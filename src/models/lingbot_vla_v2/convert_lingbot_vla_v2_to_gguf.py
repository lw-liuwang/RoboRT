#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Convert the lingbot-vla-v2-6b (Robbyant) flow-matching VLA checkpoint into a
# RoboRT GGUF. The Qwen3-VL vision tower is NOT included here - export it
# separately with convert_lingbot_mmproj_to_gguf.py.
#
# Layout of this file:
#   token_embd            VLM token embedding [vocab, 2560]
#   vlm.blk.N.*           Qwen3-VL text tower (36 layers, QK-norm, no bias)
#   aex.blk.N.*           Qwen2-style action expert (36 layers, q/k/v bias,
#                         AdaRMSNorm time-conditioned norms, per-layer MoE)
#   aex.blk.N.experts_*   3D fused expert weights [E, inter, hidden] (raw,
#                         no permute - matches ggml_mul_mat_id layout)
#   action_*_proj / time  flow-matching heads
#   prefix_query_*        precomputed [8, 2560] query token banks (mean over
#                         the 32-slot groups of the align_embs banks, fused
#                         through current/future_shared_task_proj)
#   state_q01/...         55-dim bounds_99_woclip norm stats
#
# Usage:
#   python convert_lingbot_vla_v2_to_gguf.py \
#     --ckpt /path/to/lingbot-vla-v2-6b \
#     --stats /path/to/norm_stats/robotwin.json \
#     --out lingbot-vla-v2-6b-f32.gguf [--dtype bf16]

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "third_party" / "gguf-py"))
import gguf

ARCH = "lingbot_vla_v2"
KV = lambda name: f"{ARCH}.{name}"

PFX_VLM_CANDIDATES = [
    "model.qwenvl_with_expert.qwenvl.model.language_model",
    "qwenvl_with_expert.qwenvl.model.language_model",
]
PFX_AEX_CANDIDATES = [
    "model.qwenvl_with_expert.qwen_expert.model",
    "qwenvl_with_expert.qwen_expert.model",
]
PFX_PROJ_CANDIDATES = ["model", ""]

# ---- architecture constants (lingbot-vla-v2-6b / robotwin config) ----
# VLM = Qwen3-VL-4B-Instruct text tower
VLM = dict(hidden=2560, n_q_heads=32, n_kv_heads=8, head_dim=128, intermediate=9728)
# action expert = Qwen2-style 768-wide decoder with per-layer token MoE
AEX = dict(hidden=768, n_q_heads=32, n_kv_heads=8, head_dim=128, norm_eps=1e-6)
MOE = dict(n_experts=32, top_k=4, intermediate=512, shared_intermediate=704,
           routed_scaling_factor=4.0, router="sigmoid")
FLOW = dict(chunk_size=50, num_steps=10, max_state_dim=55, max_action_dim=55,
            time_dim=768, min_period=4e-3, max_period=4.0)
PREFIX = dict(num_task_tokens=8, deepstack_layers=[5, 11, 17],
              rope_theta=5e6, rms_norm_eps=1e-6)
# canonical 55-dim state/action layout (robotwin joints: arm.position:14,
# end.position:14, effector.position:2 - only arm+effector are live)
STATE_LAYOUT = dict(arm_offset=0, arm_dim=12,
                    effector_offset=28, effector_dim=2)
# Raw RoboTwin observation.state/action order (robotwin.yaml robot_config
# origin_keys): [armL 6, gripL 1, armR 6, gripR 1].  Map raw index ->
# canonical 55-dim offset: armL->arm[0:6], armR->arm[6:12], grippers->eff[0:2].
RAW_TO_CANON = [0, 1, 2, 3, 4, 5, 28, 6, 7, 8, 9, 10, 11, 29]


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).contiguous().cpu().numpy()


def _join_key(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _pick_prefix(keys: set[str], candidates: list[str], probe_suffix: str) -> str:
    for prefix in candidates:
        probe = _join_key(prefix, probe_suffix) if probe_suffix else prefix
        if probe in keys:
            return prefix
    raise SystemExit(f"cannot resolve checkpoint prefix for {probe_suffix or candidates[0]!r}")


class ShardReader:
    """Read tensors from a sharded safetensors checkpoint."""

    def __init__(self, ckpt: Path):
        idx_path = ckpt / "model.safetensors.index.json"
        single = ckpt / "model.safetensors"
        self.handles = {}
        if idx_path.is_file():
            idx = json.loads(idx_path.read_text())
            self.weight_map = idx["weight_map"]
            self.shards = sorted(set(self.weight_map.values()))
        elif single.is_file():
            self.shards = ["model.safetensors"]
            self.weight_map = None
        else:
            raise SystemExit(f"no safetensors found in {ckpt}")

    def open(self, ckpt: Path):
        for s in self.shards:
            self.handles[s] = safe_open(str(ckpt / s), framework="pt")

    def keys(self) -> set[str]:
        if self.weight_map is not None:
            return set(self.weight_map.keys())
        k = set()
        for h in self.handles.values():
            k |= set(h.keys())
        return k

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name] if self.weight_map is not None else self.shards[0]
        return self.handles[shard].get_tensor(name)

    def get_shape(self, name: str) -> list[int]:
        shard = self.weight_map[name] if self.weight_map is not None else self.shards[0]
        return list(self.handles[shard].get_slice(name).get_shape())


def _load_norm_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load observation.state / action q01/q99 and scatter them into the
    canonical 55-dim layout (identity stats on dead dims)."""
    dim = FLOW["max_state_dim"]
    state_q01 = -np.ones(dim, dtype=np.float32)
    state_q99 = np.ones(dim, dtype=np.float32)
    action_q01 = -np.ones(dim, dtype=np.float32)
    action_q99 = np.ones(dim, dtype=np.float32)

    def _fill(prefix: str, out01: np.ndarray, out99: np.ndarray):
        for key, offset, length in [
            ("arm.position", STATE_LAYOUT["arm_offset"], STATE_LAYOUT["arm_dim"]),
            ("effector.position", STATE_LAYOUT["effector_offset"], STATE_LAYOUT["effector_dim"]),
        ]:
            stats = d[f"{prefix}.{key}"]
            out01[offset:offset + length] = np.asarray(stats["q01"], dtype=np.float32)
            out99[offset:offset + length] = np.asarray(stats["q99"], dtype=np.float32)

    if not stats_path.is_file():
        print(f"  stats: {stats_path} not found - using identity stats (unnormalized!)")
        return state_q01, state_q99, action_q01, action_q99

    d = json.loads(stats_path.read_text())["norm_stats"]
    _fill("observation.state", state_q01, state_q99)
    _fill("action", action_q01, action_q99)
    print(f"  stats: loaded bounds_99_woclip q01/q99 from {stats_path.name}")
    return state_q01, state_q99, action_q01, action_q99


def _add_kv(writer: gguf.GGUFWriter, vlm_causal: bool = True,
            use_shared_expert_gate: bool = False) -> None:
    writer.add_string(KV("architecture"), ARCH)
    # vlm_causal=1 -> causal prefix mask (robotwin.yaml); 0 -> fully
    # bidirectional prefix (make_att_2d_masks with all-zero att_masks).
    writer.add_uint32(KV("vlm_causal"), 1 if vlm_causal else 0)
    writer.add_uint32(KV("hidden"), VLM["hidden"])
    writer.add_uint32(KV("intermediate"), VLM["intermediate"])
    writer.add_uint32(KV("n_q_heads"), VLM["n_q_heads"])
    writer.add_uint32(KV("n_kv_heads"), VLM["n_kv_heads"])
    writer.add_uint32(KV("head_dim"), VLM["head_dim"])
    writer.add_uint32(KV("n_layers"), N_LAYERS)
    writer.add_uint32(KV("vocab_size"), VOCAB)
    writer.add_uint32(KV("expert_h"), AEX["hidden"])
    writer.add_uint32(KV("expert_n_q_heads"), AEX["n_q_heads"])
    writer.add_uint32(KV("expert_n_kv_heads"), AEX["n_kv_heads"])
    writer.add_uint32(KV("expert_head_dim"), AEX["head_dim"])
    writer.add_float32(KV("expert_norm_eps"), AEX["norm_eps"])
    writer.add_uint32(KV("moe_n_experts"), MOE["n_experts"])
    writer.add_uint32(KV("moe_top_k"), MOE["top_k"])
    writer.add_uint32(KV("moe_intermediate"), MOE["intermediate"])
    writer.add_uint32(KV("moe_shared_intermediate"), MOE["shared_intermediate"])
    # use_shared_expert_gate=1 -> sigmoid(Linear(hidden->1)) gates the shared
    # expert output (Qwen2TokenMoeBlock with use_shared_expert_gate=True);
    # 0 -> shared expert added ungated (robotwin.yaml sets it false, so the
    # checkpoint has no shared_expert_gate weights at all).
    writer.add_uint32(KV("moe_use_shared_expert_gate"), 1 if use_shared_expert_gate else 0)
    writer.add_float32(KV("moe_routed_scaling_factor"), MOE["routed_scaling_factor"])
    writer.add_string(KV("moe_router_activation"), MOE["router"])
    writer.add_uint32(KV("chunk_size"), FLOW["chunk_size"])
    writer.add_uint32(KV("num_steps"), FLOW["num_steps"])
    writer.add_uint32(KV("max_state_dim"), FLOW["max_state_dim"])
    writer.add_uint32(KV("max_action_dim"), FLOW["max_action_dim"])
    writer.add_uint32(KV("time_dim"), FLOW["time_dim"])
    writer.add_float64(KV("min_period"), FLOW["min_period"])
    writer.add_float64(KV("max_period"), FLOW["max_period"])
    writer.add_uint32(KV("num_task_tokens"), PREFIX["num_task_tokens"])
    writer.add_array(KV("deepstack_layers"), PREFIX["deepstack_layers"])
    writer.add_float64(KV("rope_theta"), PREFIX["rope_theta"])
    writer.add_float32(KV("rms_norm_eps"), PREFIX["rms_norm_eps"])
    # canonical state/action layout (dead dims keep identity norm stats)
    writer.add_uint32(KV("state_arm_offset"), STATE_LAYOUT["arm_offset"])
    writer.add_uint32(KV("state_arm_dim"), STATE_LAYOUT["arm_dim"])
    writer.add_uint32(KV("state_effector_offset"), STATE_LAYOUT["effector_offset"])
    writer.add_uint32(KV("state_effector_dim"), STATE_LAYOUT["effector_dim"])
    # raw (robot) dim -> canonical offset; state and action share the layout.
    writer.add_array(KV("state_raw_map"), RAW_TO_CANON)
    writer.add_array(KV("action_raw_map"), RAW_TO_CANON)
    writer.add_string(KV("state_norm_mode"), "bounds_99_woclip")
    writer.add_string(KV("action_norm_mode"), "bounds_99_woclip")


N_LAYERS = 36
VOCAB = 151936


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert lingbot-vla-v2-6b to RoboRT GGUF")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="lingbot-vla-v2-6b checkpoint directory")
    ap.add_argument("--stats", type=Path, default=None,
                    help="norm_stats json (e.g. assets/norm_stats/robotwin.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output GGUF path (default: <ckpt>/lingbot-vla-v2-6b.gguf)")
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32",
                    help="Weight dtype for linear weights (norms/biases/router stay F32). Default f32.")
    ap.add_argument("--vlm-causal", dest="vlm_causal", action="store_true", default=True,
                    help="Causal prefix attention (robotwin.yaml default).")
    ap.add_argument("--no-vlm-causal", dest="vlm_causal", action="store_false",
                    help="Fully bidirectional prefix attention (vlm_causal=false configs).")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out = (args.out or ckpt / "lingbot-vla-v2-6b.gguf").resolve()
    stats_path = args.stats

    rd = ShardReader(ckpt)
    rd.open(ckpt)
    keys = rd.keys()

    def get(name: str) -> torch.Tensor:
        if name not in keys:
            raise SystemExit(f"missing tensor {name}")
        return rd.get(name)

    def shape(name: str) -> list[int]:
        if name not in keys:
            raise SystemExit(f"missing tensor {name}")
        return rd.get_shape(name)

    pfx_vlm = _pick_prefix(keys, PFX_VLM_CANDIDATES, "layers.0.self_attn.q_proj.weight")
    pfx_aex = _pick_prefix(keys, PFX_AEX_CANDIDATES, "layers.0.self_attn.q_proj.weight")
    pfx_proj = _pick_prefix(keys, PFX_PROJ_CANDIDATES, "action_in_proj.weight")

    def _maxlayer(pfx: str) -> int:
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try:
                    m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError:
                    pass
        return m + 1

    global N_LAYERS, VOCAB
    n_layers_vlm = _maxlayer(f"{pfx_vlm}.layers.")
    n_layers_aex = _maxlayer(f"{pfx_aex}.layers.")
    if n_layers_vlm <= 0:
        raise SystemExit("cannot find VLM layers in checkpoint")
    if n_layers_vlm != n_layers_aex:
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex}")
    N_LAYERS = n_layers_vlm

    emb = f"{pfx_vlm}.embed_tokens.weight"
    VOCAB = shape(emb)[0]

    # ---- shape validation ----
    q0 = shape(f"{pfx_vlm}.layers.0.self_attn.q_proj.weight")
    kv0 = shape(f"{pfx_vlm}.layers.0.self_attn.k_proj.weight")
    gate0 = shape(f"{pfx_vlm}.layers.0.mlp.gate_proj.weight")
    if q0[1] != VLM["hidden"]:
        raise SystemExit(f"VLM hidden mismatch: cfg={VLM['hidden']} ckpt={q0[1]}")
    if q0[0] != VLM["n_q_heads"] * VLM["head_dim"]:
        raise SystemExit(f"VLM q rows {q0[0]} != {VLM['n_q_heads']*VLM['head_dim']}")
    if kv0[0] != VLM["n_kv_heads"] * VLM["head_dim"]:
        raise SystemExit(f"VLM k rows {kv0[0]} != {VLM['n_kv_heads']*VLM['head_dim']}")
    if gate0[0] != VLM["intermediate"]:
        raise SystemExit(f"VLM intermediate mismatch: cfg={VLM['intermediate']} ckpt={gate0[0]}")
    if shape(f"{pfx_vlm}.layers.0.self_attn.q_norm.weight") != [VLM["head_dim"]]:
        raise SystemExit("VLM q_norm shape mismatch (expected Qwen3 QK-norm [head_dim])")

    aq0 = shape(f"{pfx_aex}.layers.0.self_attn.q_proj.weight")
    if aq0[1] != AEX["hidden"]:
        raise SystemExit(f"expert hidden mismatch: cfg={AEX['hidden']} ckpt={aq0[1]}")
    if aq0[0] != AEX["n_q_heads"] * AEX["head_dim"]:
        raise SystemExit(f"expert q rows {aq0[0]} != {AEX['n_q_heads']*AEX['head_dim']}")
    # AdaRMSNorm gamma/beta project cond(=768 time emb) -> expert hidden
    g0 = shape(f"{pfx_aex}.layers.0.input_layernorm.gamma.weight")
    if g0 != [AEX["hidden"], FLOW["time_dim"]]:
        raise SystemExit(f"expert AdaRMS gamma shape {g0} != [{AEX['hidden']}, {FLOW['time_dim']}]")
    # fused experts: [E, inter, hidden] / [E, hidden, inter] (no .weight suffix in ckpt)
    e_gate = shape(f"{pfx_aex}.layers.0.mlp.experts.gate_proj")
    e_down = shape(f"{pfx_aex}.layers.0.mlp.experts.down_proj")
    if e_gate != [MOE["n_experts"], MOE["intermediate"], AEX["hidden"]]:
        raise SystemExit(f"experts.gate_proj shape {e_gate} != "
                         f"[{MOE['n_experts']}, {MOE['intermediate']}, {AEX['hidden']}]")
    if e_down != [MOE["n_experts"], AEX["hidden"], MOE["intermediate"]]:
        raise SystemExit(f"experts.down_proj shape {e_down} != "
                         f"[{MOE['n_experts']}, {AEX['hidden']}, {MOE['intermediate']}]")
    sh_gate = shape(f"{pfx_aex}.layers.0.mlp.shared_expert.gate_proj.weight")
    if sh_gate != [MOE["shared_intermediate"], AEX["hidden"]]:
        raise SystemExit(f"shared_expert gate shape {sh_gate} != "
                         f"[{MOE['shared_intermediate']}, {AEX['hidden']}]")
    # use_shared_expert_gate detection: the gate Linear only exists when the
    # training config enabled it (robotwin.yaml: use_shared_expert_gate=false).
    use_seg = f"{pfx_aex}.layers.0.mlp.shared_expert_gate.weight" in keys

    print(f"resolved: VLM hidden={VLM['hidden']} layers={N_LAYERS} vocab={VOCAB} "
          f"expert_h={AEX['hidden']} moe={MOE['n_experts']}x top{MOE['top_k']} "
          f"chunk={FLOW['chunk_size']} steps={FLOW['num_steps']} dtype={args.dtype} "
          f"shared_expert_gate={'on' if use_seg else 'off (ungated)'}")

    if stats_path is None:
        for cand in [ckpt / "norm_stats.json", ckpt.parent / "norm_stats.json"]:
            if cand.is_file():
                stats_path = cand
                break
    state_q01, state_q99, action_q01, action_q99 = _load_norm_stats(
        stats_path) if stats_path is not None else _load_norm_stats(Path("__missing__"))

    def add(name: str, t: torch.Tensor, force_f32: bool = False):
        if args.dtype == "bf16" and not force_f32:
            writer.add_tensor(name, _bf16_to_u16_bytes(t),
                              raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, _f32_np(t), raw_dtype=gguf.GGMLQuantizationType.F32)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    _add_kv(writer, vlm_causal=args.vlm_causal, use_shared_expert_gate=use_seg)
    n_tensors = 0

    # ---- VLM token embedding + final norm ----
    add("token_embd.weight", get(emb))
    add("vlm.output_norm.weight", get(f"{pfx_vlm}.norm.weight"), force_f32=True)
    n_tensors += 2

    # ---- VLM (Qwen3-VL text) layers ----
    for i in range(N_LAYERS):
        src = f"{pfx_vlm}.layers.{i}"
        dst = f"vlm.blk.{i}"
        add(f"{dst}.attn_norm.weight", get(f"{src}.input_layernorm.weight"), force_f32=True)
        add(f"{dst}.attn_q.weight", get(f"{src}.self_attn.q_proj.weight"))
        add(f"{dst}.attn_q_norm.weight", get(f"{src}.self_attn.q_norm.weight"), force_f32=True)
        add(f"{dst}.attn_k.weight", get(f"{src}.self_attn.k_proj.weight"))
        add(f"{dst}.attn_k_norm.weight", get(f"{src}.self_attn.k_norm.weight"), force_f32=True)
        add(f"{dst}.attn_v.weight", get(f"{src}.self_attn.v_proj.weight"))
        add(f"{dst}.attn_o.weight", get(f"{src}.self_attn.o_proj.weight"))
        add(f"{dst}.ffn_norm.weight", get(f"{src}.post_attention_layernorm.weight"), force_f32=True)
        add(f"{dst}.ffn_gate.weight", get(f"{src}.mlp.gate_proj.weight"))
        add(f"{dst}.ffn_up.weight", get(f"{src}.mlp.up_proj.weight"))
        add(f"{dst}.ffn_down.weight", get(f"{src}.mlp.down_proj.weight"))
        n_tensors += 12

    # ---- expert (Qwen2 + AdaRMS + MoE) layers ----
    for i in range(N_LAYERS):
        src = f"{pfx_aex}.layers.{i}"
        dst = f"aex.blk.{i}"
        # AdaRMSNorm: weight + FiLM gamma/beta (time-cond)
        for norm_src, norm_dst in [("input_layernorm", "attn_norm"),
                                   ("post_attention_layernorm", "ffn_norm")]:
            add(f"{dst}.{norm_dst}.weight", get(f"{src}.{norm_src}.weight"), force_f32=True)
            add(f"{dst}.{norm_dst}.gamma_w", get(f"{src}.{norm_src}.gamma.weight"))
            add(f"{dst}.{norm_dst}.gamma_b", get(f"{src}.{norm_src}.gamma.bias"), force_f32=True)
            add(f"{dst}.{norm_dst}.beta_w", get(f"{src}.{norm_src}.beta.weight"))
            add(f"{dst}.{norm_dst}.beta_b", get(f"{src}.{norm_src}.beta.bias"), force_f32=True)
        # attention with q/k/v bias
        add(f"{dst}.attn_q.weight", get(f"{src}.self_attn.q_proj.weight"))
        add(f"{dst}.attn_q.bias", get(f"{src}.self_attn.q_proj.bias"), force_f32=True)
        add(f"{dst}.attn_k.weight", get(f"{src}.self_attn.k_proj.weight"))
        add(f"{dst}.attn_k.bias", get(f"{src}.self_attn.k_proj.bias"), force_f32=True)
        add(f"{dst}.attn_v.weight", get(f"{src}.self_attn.v_proj.weight"))
        add(f"{dst}.attn_v.bias", get(f"{src}.self_attn.v_proj.bias"), force_f32=True)
        add(f"{dst}.attn_o.weight", get(f"{src}.self_attn.o_proj.weight"))
        # MoE: router gate + e_score_correction_bias + fused routed experts + shared expert
        add(f"{dst}.moe_gate.weight", get(f"{src}.mlp.gate.weight"), force_f32=True)
        add(f"{dst}.experts_bias", get(f"{src}.mlp.e_score_correction_bias"), force_f32=True)
        add(f"{dst}.experts_gate.weight", get(f"{src}.mlp.experts.gate_proj"))
        add(f"{dst}.experts_up.weight", get(f"{src}.mlp.experts.up_proj"))
        add(f"{dst}.experts_down.weight", get(f"{src}.mlp.experts.down_proj"))
        add(f"{dst}.shared_expert_gate.weight", get(f"{src}.mlp.shared_expert.gate_proj.weight"))
        add(f"{dst}.shared_expert_up.weight", get(f"{src}.mlp.shared_expert.up_proj.weight"))
        add(f"{dst}.shared_expert_down.weight", get(f"{src}.mlp.shared_expert.down_proj.weight"))
        # optional sigmoid gate on the shared output: only exported when the
        # checkpoint contains it (use_shared_expert_gate=True in training).
        if use_seg:
            add(f"{dst}.shared_expert_router.weight", get(f"{src}.mlp.shared_expert_gate.weight"), force_f32=True)
            n_tensors += 27
        else:
            n_tensors += 26

    # ---- expert final norm (plain RMSNorm: final_norm_adanorm=False) ----
    add("aex.output_norm.weight", get(f"{pfx_aex}.norm.weight"), force_f32=True)
    n_tensors += 1

    # ---- flow-matching heads ----
    for suf in ["state_proj.weight", "state_proj.bias",
                "action_in_proj.weight", "action_in_proj.bias",
                "action_out_proj.weight", "action_out_proj.bias",
                "action_time_mlp_in.weight", "action_time_mlp_in.bias",
                "action_time_mlp_out.weight", "action_time_mlp_out.bias"]:
        add(suf, get(_join_key(pfx_proj, suf)), force_f32=suf.endswith("bias"))
        n_tensors += 1

    # ---- prefix query token banks ----
    # robotwin inference path: each query bank is the mean over the
    # num_task_tokens groups of a [num_backbone_tokens, 2560] align_embs bank,
    # then fused through the shared task projection:
    #   current = current_shared_task_proj(
    #       cat([mean(depth_align_embs), mean(current_video_align_embs)], -1))
    #   future  = future_shared_task_proj(
    #       cat([mean(future_depth_align_embs), mean(future_video_align_embs)], -1))
    # (enabled by align_params.video: use_patch_loss + use_current_patch_loss +
    #  use_current_shared_task_proj + use_shared_future_task_proj +
    #  share_future_depth_query, all true in robotwin.yaml).
    n_tok = PREFIX["num_task_tokens"]

    def mean_groups(name: str) -> torch.Tensor:
        bank = get(_join_key(pfx_proj, name)).to(torch.float32)
        assert bank.dim() == 2 and bank.shape[0] % n_tok == 0, \
            f"{name} shape {tuple(bank.shape)}"
        return bank.view(n_tok, bank.shape[0] // n_tok, bank.shape[1]).mean(dim=1)

    def fused_query(proj_name: str, bank_a: str, bank_b: str) -> torch.Tensor:
        w = get(_join_key(pfx_proj, f"{proj_name}.weight")).to(torch.float32)
        b = get(_join_key(pfx_proj, f"{proj_name}.bias")).to(torch.float32)
        x = torch.cat([mean_groups(bank_a), mean_groups(bank_b)], dim=-1)
        assert x.shape == (n_tok, w.shape[1]), \
            f"{proj_name}: input {tuple(x.shape)} vs weight {tuple(w.shape)}"
        return x @ w.T + b

    add("prefix_query_current", fused_query(
        "current_shared_task_proj", "depth_align_embs", "current_video_align_embs"),
        force_f32=True)
    add("prefix_query_future", fused_query(
        "future_shared_task_proj", "future_depth_align_embs", "future_video_align_embs"),
        force_f32=True)
    n_tensors += 2

    # ---- norm stats ----
    for name, arr in [("state_q01", state_q01), ("state_q99", state_q99),
                      ("action_q01", action_q01), ("action_q99", action_q99)]:
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        n_tensors += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"tensors: {n_tensors}, dtype: {args.dtype}")
    print(f"wrote {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB)")
    print("note: the Qwen3-VL vision tower is NOT in this file - produce the mmproj "
          "GGUF separately with convert_lingbot_mmproj_to_gguf.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
