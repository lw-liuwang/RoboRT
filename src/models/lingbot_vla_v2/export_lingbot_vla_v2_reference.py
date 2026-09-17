#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dump HuggingFace lingbot-vla-v2 reference intermediates to an .npz for
C++ (RoboRT) numerical alignment.

Mirrors deploy/lingbot_vla_v2_policy.py's load path (lingbotvla_cli.yaml ==
configs/vla/robotwin/robotwin.yaml) and select_action flow, with:
  - eager joint attention (config.attention_implementation='eager')
  - sdpa ViT (flash_attn-free: set --vit-attn sdpa|eager)
  - deterministic inputs: seeded RGB images / raw state / language prompt /
    denoiser noise
  - fixed seed; TF32 disabled by default for a clean fp32/bf16 gold trace

Captured stages (all fp32 numpy, batch dim squeezed):
  img/pixel_values, img/grid_thw        processor output entering the ViT
  img/clip_out, img/deepstack{0,1,2}   merger output + deepstack ViT features
  query/current, query/future           fused prefix query banks [8, 2560]
  lang/tokens, lang/embs
  prefix/embs [P,2560], pad/att masks, pos [3,P], visual_pos, deepstack rows
  prefix/k_L{i}, prefix/v_L{i}          roped K / raw V cached per layer
  prefix/out_vlm [P,2560]               final RMSNorm output (prefix pass)
  prefix/hid_in_L{0,1,2,35}             VLM layer inputs (deepstack check)
  s{t}/time_emb [768], suffix_embs [51,768], x_in/v [50,55], suffix_out
  s0/att2d [51,P+51], s0/pos [3,51], s0/k_L{i}+v_L{i} (prefix+suffix concat)
  s0/hid_in_L{i} [51,768]               expert tower layer inputs
  final/x [50,55]                       denormalized-domain x_final
  final/raw_arm [50,14], raw_effector [50,2]   unapply() output
  state/raw [14], state/canon [55], noise [50,55]

Usage (inside lw_robort_eval container, /workspace/.venv):
  python export_lingbot_vla_v2_reference.py \
    --ckpt  /mount/lw/github/RoboRT/local/models/lingbot-vla-v2-6b \
    --qwen3vl /mount/lw/github/RoboRT/local/models/Qwen3-VL-4B-Instruct \
    --out /tmp/lingbot_ref.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    here = Path(__file__).resolve().parent
    robo = here.parents[2]
    p.add_argument("--hf-root", type=Path, default=robo / "local" / "lingbot-vla-v2",
                   help="lingbot-vla-v2 source repo (importable lingbotvla package)")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="lingbot-vla-v2-6b checkpoint dir (sharded safetensors)")
    p.add_argument("--qwen3vl", type=Path, required=True,
                   help="Qwen3-VL-4B-Instruct base dir (config + processor + tokenizer)")
    p.add_argument("--train-config", type=Path,
                   default=robo / "local" / "lingbot-vla-v2" / "configs" / "vla" / "robotwin" / "robotwin.yaml")
    p.add_argument("--robot-config", type=Path,
                   default=robo / "local" / "lingbot-vla-v2" / "configs" / "robot_configs" / "robotwin.yaml")
    p.add_argument("--norm-stats", type=Path,
                   default=robo / "local" / "lingbot-vla-v2" / "assets" / "norm_stats" / "robotwin.json")
    p.add_argument("--out", type=Path, required=True, help="output .npz path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="model dtype (bf16 mirrors deploy default; fp32 needs ~26GB)")
    p.add_argument("--vit-attn", choices=["sdpa", "eager"], default="sdpa",
                   help="ViT attention implementation (never flash)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls")
    p.add_argument("--prompt", default="pick up the red cube and place it on the shelf")
    p.add_argument("--steps-per-layer-hidden", action="store_true",
                   help="also dump per-layer hidden inputs for every denoise step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hf_root = args.hf_root.resolve()
    sys.path.insert(0, str(hf_root))

    import yaml
    from transformers import AutoConfig

    # ── training-only dep stubs ──────────────────────────────────────────
    # lingbotvla's import graph pulls in training-only packages that are not
    # installed in this inference venv.  Stub the missing ones: imports
    # succeed, any actual call raises.  Modules that ARE installed are never
    # shadowed (availability is probed before the finder is installed).
    def _install_training_dep_stubs() -> None:
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import types

        roots = ("torchdata", "datasets", "pyarrow", "av", "mlflow",
                 "peft", "numpydantic", "trimesh", "lerobot")
        missing = [r for r in roots if importlib.util.find_spec(r) is None]
        if not missing:
            return

        class _LazyStub:
            def __init__(self, name: str) -> None:
                object.__setattr__(self, "_name", name)

            def __getattr__(self, item: str) -> "_LazyStub":
                return _LazyStub(f"{self._name}.{item}")

            def __call__(self, *a, **k):  # pragma: no cover - never at inference
                raise ModuleNotFoundError(
                    f"training-only stub for {self._name} called")

            # allow "class X(StubbedBase):" at import time
            def __mro_entries__(self, bases):  # noqa: N805
                return (object,)

        class _StubModule(types.ModuleType):
            def __getattr__(self, item: str):  # noqa: D105
                if item.startswith("__"):
                    raise AttributeError(item)
                return _LazyStub(f"{self.__name__}.{item}")

        class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in missing:
                    return importlib.machinery.ModuleSpec(
                        fullname, self, is_package=True)
                return None

            def create_module(self, spec):
                return _StubModule(spec.name)

            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, _StubFinder())
        print(f"[stubs] training-only modules stubbed: {missing}")

    _install_training_dep_stubs()

    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest" if not args.tf32 else "high")
    device = torch.device(args.device)

    # ── config assembly (mirrors LingbotVLAv2Server.load_vla) ─────────────
    apply_lingbot_qwen3_vl_patch()

    tc = yaml.safe_load(args.train_config.read_text())
    # repo train-config yaml stores joints/norm_type entries as raw dicts; the
    # training pipeline (lingbotvla_cli.yaml) quotes them as strings which
    # FeatureInfo.update_info / get_normalizer expect (ast.literal_eval)
    for _dk in ("joints", "norm_type"):
        if isinstance(tc["data"].get(_dk), list):
            tc["data"][_dk] = [j if isinstance(j, str) else str(j)
                               for j in tc["data"][_dk]]
    tm = dict(tc["model"])
    tm.update(tc["train"])
    config = LingbotVLAV2Config(**tm)
    for k, v in tm.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    config.tokenizer_path = str(args.qwen3vl)
    config.attention_implementation = "eager"   # joint attention (deploy eval)
    config.vit_attn_implementation = args.vit_attn
    config.use_cache = True

    qwen_config = AutoConfig.from_pretrained(str(args.qwen3vl))
    # merge_qwen_config (deploy policy): text keys + vision_config
    cfg_dict = qwen_config.to_dict()
    text_keys = {"hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
                 "rope_theta", "vocab_size", "max_position_embeddings",
                 "hidden_act", "tie_word_embeddings", "tokenizer_path"}
    for key in text_keys:
        if key in cfg_dict.get("text_config", {}):
            setattr(config, key, cfg_dict["text_config"][key])
        elif key in cfg_dict:
            setattr(config, key, cfg_dict[key])
    config.vision_config = qwen_config.vision_config
    if tm.get("vocab_size", 0):
        config.vocab_size = tm["vocab_size"]

    processor = build_processor(str(args.qwen3vl))
    print(f"[config] hidden={config.max_state_dim} steps={config.num_steps} "
          f"chunk={config.chunk_size} attn={config.attention_implementation} "
          f"vit={config.vit_attn_implementation} use_moe={config.use_moe} "
          f"shared_gate={getattr(config, 'use_shared_expert_gate', True)}")

    # ── model + weights ───────────────────────────────────────────────────
    policy = LingbotVlaV2Policy(config, eval=True)
    from safetensors.torch import load_file
    merged = {}
    for shard in sorted(args.ckpt.glob("*.safetensors")):
        merged.update(load_file(str(shard)))
    missing, unexpected = policy.load_state_dict(merged, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected checkpoint keys: {unexpected[:8]}")
    if missing:
        raise SystemExit(f"missing checkpoint keys: {missing[:8]}")
    del merged
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    policy = policy.to(dtype=dtype, device=device).eval()
    M = policy.model
    # policy.__init__ sets "high"; re-apply our precision choice after it
    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision("high")

    # the joint/ViT dispatch never uses flash here; text config value is inert
    M.qwenvl_with_expert.qwenvl.config.text_config._attn_implementation = args.vit_attn
    M.qwenvl_with_expert.qwenvl.config.vision_config._attn_implementation = args.vit_attn

    # ── CPU MoE path ─────────────────────────────────────────────────────
    # lingbotvla's fused/robby MoE kernels are CUDA(triton)-only; on CPU
    # replace the non-EP fused path with a faithful pure-torch replica
    # (silu(gate)*up -> down, routing weight applied to the expert output).
    # Computed in fp32 — strictly more accurate than the GPU bf16 fused path.
    if device.type != "cuda":
        import torch.nn.functional as F_
        import lingbotvla.ops.fused_moe as fused_moe_mod

        def _cpu_fused_moe_forward(module, num_experts, routing_weights,
                                   selected_experts, hidden_states,
                                   fc1_1_weight, fc1_2_weight, fc2_weight):
            T, D = hidden_states.shape
            out = torch.zeros((T, D), dtype=torch.float32,
                              device=hidden_states.device)
            rw = routing_weights.float()
            for e in range(num_experts):
                tok, slot = (selected_experts == e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                x = hidden_states[tok]
                h = F_.silu(F_.linear(x, fc1_1_weight[e].float())) \
                    * F_.linear(x, fc1_2_weight[e].float())
                y = F_.linear(h, fc2_weight[e].float())
                out.index_add_(0, tok, y * rw[tok, slot].unsqueeze(-1))
            return out.to(hidden_states.dtype)

        fused_moe_mod.fused_moe_forward = _cpu_fused_moe_forward
        print("[moe] CPU: fused_moe_forward replaced with pure-torch fp32 path")

    # ── feature transform (mirrors server.reset) ──────────────────────────
    data_config = SimpleNamespace(**tc["data"])
    ft = FeatureTransform(str(args.robot_config), data_config, config, processor,
                          chunk_size=config.chunk_size, norm_stats_path=str(args.norm_stats))

    # ── deterministic observation ─────────────────────────────────────────
    # mirror server._prepare_model_input: uint8 HWC -> float32 CHW (256x256,
    # Resize is the identity here; values stay integral so _to_uint8 in
    # pad_and_concat is a no-op and the processor sees [0, 255] range).
    rng = np.random.default_rng(args.seed)
    cam_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    obs = {}
    for cam in cam_keys:
        hwc = torch.from_numpy(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
        obs[f"observation.images.{cam}"] = hwc.permute(2, 0, 1).float()
    raw_state = rng.uniform(-0.5, 0.5, size=14).astype(np.float32)
    obs["observation.state"] = torch.from_numpy(raw_state)
    obs["task"] = args.prompt

    batch = ft.apply(obs, policy_eval=True)
    images = batch["images"].unsqueeze(0).to(dtype=dtype, device=device)
    img_masks = batch["img_masks"].unsqueeze(0).to(device=device)
    lang_tokens = batch["lang_tokens"].unsqueeze(0).to(device=device)
    lang_masks = batch["lang_masks"].unsqueeze(0).to(device=device)
    state = batch["state"].unsqueeze(0).to(dtype=dtype, device=device)
    grid = batch["image_grid_thw"].to(device=device)

    n_lang = int(lang_masks[0].sum().item())
    n_img_tok = int((grid[:, 1] * grid[:, 2] // 4).sum().item())  # merge2
    print(f"[obs] prefix P = {2 * images.shape[1] + n_img_tok + 16 + n_lang} "
          f"(img {images.shape[1]}x{images.shape[2]} patches -> {n_img_tok} tok, "
          f"queries 16, lang {n_lang})")

    noise = torch.randn(1, config.n_action_steps, config.max_action_dim,
                        device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(args.seed))

    # ── capture machinery ─────────────────────────────────────────────────
    ref: dict[str, np.ndarray] = {}

    def put(key: str, t):
        if t is None:
            return
        if torch.is_tensor(t):
            # .clone() is essential: .float()/.cpu() are no-ops for fp32 CPU
            # tensors, so without it .numpy() would alias the live storage and
            # in-place updates (e.g. x_t += dt*v_t mutating the noise tensor)
            # would silently rewrite stored dumps.
            t = t.detach().clone().float().cpu().numpy()
        ref[key] = np.array(t)

    qwe = M.qwenvl_with_expert
    prog = {"step": -1, "pass": -1}
    layer_stash = {}

    # pixel values + clip output / deepstack features
    orig_embed_image = qwe.embed_image
    def embed_image(image, image_grid_thw):
        out = orig_embed_image(image, image_grid_thw)
        put("img/clip_out", out[0])                 # [1, n, l, 2560]
        for i, d in enumerate(out[1]):
            put(f"img/deepstack{i}", d)
        return out
    qwe.embed_image = embed_image

    # prefix embeddings / masks / positions
    orig_embed_prefix = M.embed_prefix
    def embed_prefix(img, im, lt, lm, image_grid_thw=None):
        out = orig_embed_prefix(img, im, lt, lm, image_grid_thw=image_grid_thw)
        put("prefix/embs", out[0])                  # [1, P, 2560]
        put("prefix/pad", out[1])
        put("prefix/att", out[2])
        put("prefix/pos", out[3])                   # [3, 1, P]
        put("prefix/visual_pos", out[4])
        for i, d in enumerate(out[5]):
            put(f"prefix/deepstack{i}", d)
        return out
    M.embed_prefix = embed_prefix

    # suffix embeddings (per denoise step) + raw time embedding
    orig_embed_suffix = M.embed_suffix
    def embed_suffix(st, x_t, timestep):
        prog["step"] += 1
        time_embs, suffix_embs, pad, att = orig_embed_suffix(st, x_t, timestep)
        step = prog["step"]
        put(f"s{step}/time_emb", time_embs)         # [1, 768]
        put(f"s{step}/suffix_embs", suffix_embs)    # [1, 51, 768]
        put(f"s{step}/x_in", x_t)
        return time_embs, suffix_embs, pad, att
    M.embed_suffix = embed_suffix

    # KV cache fill (prefix) / read (suffix concat)
    orig_hkv = qwe.handle_kv_cache
    def handle_kv_cache(key_states, value_states, layer_idx, past_key_values=None,
                        use_cache=None, fill_kv_cache=None):
        k, v, pkv = orig_hkv(key_states, value_states, layer_idx,
                             past_key_values=past_key_values, use_cache=use_cache,
                             fill_kv_cache=fill_kv_cache)
        if fill_kv_cache:
            put(f"prefix/k_L{layer_idx}", k)        # roped K [1, P, 1024]
            put(f"prefix/v_L{layer_idx}", v)        # raw V
        elif prog["step"] == 0:
            put(f"s0/k_L{layer_idx}", k)            # concat [1, P+51, 1024]
            put(f"s0/v_L{layer_idx}", v)
        return k, v, pkv
    qwe.handle_kv_cache = handle_kv_cache

    # per-layer hidden inputs via pre-hooks (layer called twice per pass:
    # compute_kqv then output_atten - record the first call only)
    def make_pre_hook(tower: str, idx: int):
        def hook(module, fn_args, fn_kwargs):
            p = prog["pass"]
            if p < 0:
                return
            if p > 1 and not args.steps_per_layer_hidden:
                return  # default: hidden dumps only for prefix + first step
            key = (tower, p, idx)
            if key in layer_stash:
                return
            h = fn_args[0] if fn_args else fn_kwargs.get("hidden_states")
            layer_stash[key] = h.detach().float().cpu()
        return hook

    n_vlm_layers = len(qwe.qwenvl.model.language_model.layers)
    n_ex_layers = len(qwe.qwen_expert.model.layers)
    hid_vlm_dump = ({0, 1, 2, n_vlm_layers - 1} if not args.steps_per_layer_hidden
                    else set(range(n_vlm_layers)))
    for i in range(n_vlm_layers):
        if i in hid_vlm_dump:
            qwe.qwenvl.model.language_model.layers[i].register_forward_pre_hook(
                make_pre_hook("vlm", i), with_kwargs=True)
    for i in range(n_ex_layers):
        qwe.qwen_expert.model.layers[i].register_forward_pre_hook(
            make_pre_hook("ex", i), with_kwargs=True)

    # joint forward wrapper: masks / positions / ada_cond / outputs
    orig_qwe_forward = qwe.forward
    def qwe_forward(attention_mask=None, position_ids=None, vlm_position_ids=None,
                    past_key_values=None, inputs_embeds=None, use_cache=None,
                    fill_kv_cache=None, ada_cond=None, visual_pos_masks=None,
                    deepstack_visual_embeds=None, **kw):
        prog["pass"] += 1
        out = orig_qwe_forward(attention_mask=attention_mask, position_ids=position_ids,
                               vlm_position_ids=vlm_position_ids,
                               past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                               use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                               ada_cond=ada_cond, visual_pos_masks=visual_pos_masks,
                               deepstack_visual_embeds=deepstack_visual_embeds, **kw)
        outputs_embeds, pkv, _ = out
        if fill_kv_cache:
            put("prefix/out_vlm", outputs_embeds[0])       # [1, P, 2560]
            put("prefix/att2d", attention_mask)            # [1, P, P] bool
        else:
            step = prog["step"]
            put(f"s{step}/suffix_out", outputs_embeds[1])  # [1, 51, 768]
            if step == 0:
                put("s0/att2d", attention_mask)            # [1, 51, P+51]
                put("s0/pos", position_ids)                # [3, 1, 51]
                put("s0/ada_cond", ada_cond)               # [1, 768]
        return out
    qwe.forward = qwe_forward

    # velocity wrapper: v_t out (x_in already captured in embed_suffix)
    orig_pv = M.predict_velocity
    def predict_velocity(st, ppm, pkv, x_t, timestep, prefix_position_ids=None):
        v = orig_pv(st, ppm, pkv, x_t, timestep, prefix_position_ids=prefix_position_ids)
        put(f"s{prog['step']}/v", v)
        return v
    M.predict_velocity = predict_velocity

    orig_sample = M.sample_actions
    def sample_actions(images_, img_masks_, lang_tokens_, lang_masks_, st,
                       noise=None, image_grid_thw=None):
        # snapshot BEFORE the call: the denoise loop mutates the noise tensor
        # in-place (x_t = noise; x_t += dt*v_t)
        put("noise", noise)
        out = orig_sample(images_, img_masks_, lang_tokens_, lang_masks_, st,
                          noise=noise, image_grid_thw=image_grid_thw)
        put("final/x", out)
        return out
    M.sample_actions = sample_actions

    # fused prefix query banks (converter precomputes the same in GGUF)
    with torch.no_grad():
        ntt = M.num_task_tokens
        def mean_groups(t):
            g = t.shape[0] // ntt
            return t.view(ntt, g, t.shape[1]).mean(dim=1)
        cur = M.current_shared_task_proj(torch.cat(
            [mean_groups(M.depth_align_embs), mean_groups(M.current_video_align_embs)], dim=-1))
        fut = M.future_shared_task_proj(torch.cat(
            [mean_groups(M.future_depth_align_embs), mean_groups(M.future_video_align_embs)], dim=-1))
        put("query/current", cur)
        put("query/future", fut)
        put("lang/embs", qwe.embed_language_tokens(lang_tokens[0][:n_lang]))
    put("lang/tokens", lang_tokens[0][:n_lang].long())
    put("state/raw", raw_state)
    put("state/canon", batch["state"].float())
    put("img/pixel_values", images[0])
    put("img/grid_thw", grid)

    # ── run ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        x_final = M.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                   noise=noise, image_grid_thw=grid)

    # denormalized / raw-domain outputs (mirrors select_action unapply)
    out_item = dict(batch)
    out_item["actions"] = x_final.squeeze(0).float().cpu()
    out_item["state"] = out_item["state"].float()
    raw = ft.unapply(out_item)
    for k, v in raw.items():
        if torch.is_tensor(v):
            put(f"final/{k}", v)

    # per-layer hidden inputs collected by pre-hooks
    for (tower, p, idx), h in layer_stash.items():
        tag = "prefix" if p == 0 else f"s{p - 1}"
        put(f"{tag}/hid_in_{'L' if tower == 'vlm' else 'E'}{idx}", h)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{k.replace("/", "."): v for k, v in ref.items()})
    total = sum(v.nbytes for v in ref.values())
    print(f"[done] {len(ref)} arrays, {total / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
