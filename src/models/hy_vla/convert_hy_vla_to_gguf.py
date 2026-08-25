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

"""Convert a HuggingFace Hy-VLA (Hy-Embodied-0.5-VLA) checkpoint into the
single-GGUF format consumed by the RoboRT ``hy_vla`` backend
(``src/models/hy_vla/hy_vla.cpp``).

The output is a self-contained file: the dual-tower VLM (text + prefix
vision + anyres vision tower + merger), the flow-matching suffix-expert
(``_v`` family only), the outer projections, the tied lm_head (used as the
text embedding table at runtime) and the action-normalization statistics.
No tokenizer or embedding table is required -- the released checkpoints tie
``lm_head`` to ``embed_tokens``, so the C++ runtime fetches text embeddings
directly from the lm_head rows (see ``fetch_rows_f32("...lm_head.weight")``).

Usage
-----
python src/models/hy_vla/convert_hy_vla_to_gguf.py \
    --ckpt /path/to/Hy-Embodied-0.5-VLA-RoboTwin \
    --out hy_vla.gguf \
    --norm-stats /path/to/norm_stats.pkl

python src/models/hy_vla/convert_hy_vla_to_gguf.py \
    --ckpt /path/to/Hy-Embodied-0.5-VLA-RoboTwin \
    --out hy_vla.gguf \
    --dtype f32

The released RoboTwin checkpoint (chunk=40 = 20 rel + 20 abs) is produced
with the defaults: ``--min-period 0.004 --max-period 4.0`` (flow periods are
hard-coded in the HF modeling code, ``modeling_hy_vla.py``) and
``max_state_dim=32 / max_action_dim=32`` read from ``config.json``.

The converted GGUF contains 1282 tensors (~9 GiB in bf16), all weights in
BF16 and the ``norm.*`` statistics in F32, matching the reference
``Hy-Embodied-0.5-VLA-RoboTwin_bf16.gguf``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "third_party" / "gguf-py"))
import gguf

ARCH = "hy_vla"
KV = lambda name: f"{ARCH}.{name}"

# HF state-dict prefixes. The released checkpoint saves the whole policy
# under a "model." wrapper; accept both forms like the pi05 converter.
PFX_EXP_CANDIDATES = [
    "model.dual_tower.expert.model",
    "dual_tower.expert.model",
]
PFX_VLM_CANDIDATES = [
    "model.dual_tower.vlm.model.language_model.model",
    "dual_tower.vlm.model.language_model.model",
]
PFX_LM_HEAD_CANDIDATES = [
    "model.dual_tower.vlm.model.language_model.lm_head.weight",
    "dual_tower.vlm.model.language_model.lm_head.weight",
]
PFX_VIS_CANDIDATES = [
    "model.dual_tower.vlm.model.visual",
    "dual_tower.vlm.model.visual",
]
PFX_PROJ_CANDIDATES = [
    "model",
    "",
]

# Per-layer suffix maps.  C++ tensor names are fixed in
# ``src/models/hy_vla/hy_vla.cpp`` (``load_weights``); the HF source names
# come from the released checkpoint (see ``hy_vla.source_tensor_names`` in
# the reference GGUF).  The MoT "_v" family is the prefix-vision / suffix
# expert path.
TEXT_LAYER_SUFFIXES = [  # 11 non-_v tensors per VLM text layer
    ("input_layernorm.weight",            "attn_norm.weight"),
    ("self_attn.q_proj.weight",           "attn_q.weight"),
    ("self_attn.k_proj.weight",           "attn_k.weight"),
    ("self_attn.v_proj.weight",           "attn_v.weight"),
    ("self_attn.o_proj.weight",           "attn_o.weight"),
    ("self_attn.query_layernorm.weight",  "attn_q_norm.weight"),
    ("self_attn.key_layernorm.weight",    "attn_k_norm.weight"),
    ("post_attention_layernorm.weight",   "ffn_norm.weight"),
    ("mlp.gate_proj.weight",              "ffn_gate.weight"),
    ("mlp.up_proj.weight",                "ffn_up.weight"),
    ("mlp.down_proj.weight",              "ffn_down.weight"),
]
V_LAYER_SUFFIXES = [  # 9 _v tensors per layer (suffix expert + prefix vision)
    ("input_layernorm_v.weight",          "attn_norm_v.weight"),
    ("self_attn.q_proj_v.weight",         "attn_q_v.weight"),
    ("self_attn.k_proj_v.weight",         "attn_k_v.weight"),
    ("self_attn.v_proj_v.weight",         "attn_v_v.weight"),
    ("self_attn.o_proj_v.weight",         "attn_o_v.weight"),
    ("post_attention_layernorm_v.weight", "ffn_norm_v.weight"),
    ("mlp_v.gate_proj.weight",            "ffn_gate_v.weight"),
    ("mlp_v.up_proj.weight",              "ffn_up_v.weight"),
    ("mlp_v.down_proj.weight",            "ffn_down_v.weight"),
]
VISION_BLOCK_SUFFIXES = [  # 12 tensors per vision block (fused qkv / ffn)
    ("norm1.weight",       "norm1.weight"),
    ("norm1.bias",         "norm1.bias"),
    ("attn.qkv.weight",    "attn_qkv.weight"),
    ("attn.qkv.bias",      "attn_qkv.bias"),
    ("attn.proj.weight",   "attn_proj.weight"),
    ("attn.proj.bias",     "attn_proj.bias"),
    ("norm2.weight",       "norm2.weight"),
    ("norm2.bias",         "norm2.bias"),
    ("mlp.fc1.weight",     "ffn_fc1.weight"),
    ("mlp.fc1.bias",       "ffn_fc1.bias"),
    ("mlp.fc2.weight",     "ffn_fc2.weight"),
    ("mlp.fc2.bias",       "ffn_fc2.bias"),
]


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    """Return the raw BF16 bits of ``t`` as a uint16 numpy array.

    Portable across torch versions: torch < 2.1 has no ``torch.uint16`` (and
    numpy has no bfloat16), so fall back to float32 with round-to-nearest-even.
    """
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    try:
        return t.view(torch.uint16).contiguous().cpu().numpy()
    except AttributeError:
        a   = t.float().contiguous().cpu().numpy().astype(np.float32, copy=False)
        u32 = a.view(np.uint32)
        lsb = (u32 >> 16) & 1
        return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.float().contiguous().cpu().numpy()


def _join_key(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _pick_prefix(keys: set[str], candidates: list[str], probe_suffix: str) -> str:
    for prefix in candidates:
        probe = _join_key(prefix, probe_suffix) if probe_suffix else prefix
        if probe in keys:
            return prefix
    raise SystemExit(f"cannot resolve checkpoint prefix for {probe_suffix or candidates[0]!r}")


def _add_one_tensor(writer: gguf.GGUFWriter, dst_name: str, t: torch.Tensor, dtype: str) -> None:
    if dtype == "f32":
        writer.add_tensor(dst_name, _f32_np(t),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    elif dtype == "bf16":
        writer.add_tensor(dst_name, _bf16_to_u16_bytes(t),
                          raw_shape=list(t.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported --dtype {dtype!r}")


def _check_shape(name: str, t: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(t.shape) != expected:
        raise SystemExit(
            f"shape mismatch for {name}: got {tuple(t.shape)}, expected {expected}\n"
            "this checkpoint does not match the released Hy-VLA architecture"
        )


def _stream_layers(writer: gguf.GGUFWriter, sf, src_pfx: str, dst_pfx: str,
                   n_layers: int, suffix_map: list[tuple[str, str]], dtype: str) -> None:
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            t = sf.get_tensor(f"{src_pfx}.layers.{i}.{src_suf}")
            _add_one_tensor(writer, f"{dst_pfx}.blk.{i}.{dst_suf}", t, dtype)


def _stream_vision(writer: gguf.GGUFWriter, sf, vis_pfx: str,
                   n_vision_layers: int, dtype: str) -> None:
    vt = f"{vis_pfx}.vision_tower"
    pe_w = sf.get_tensor(f"{vt}.patch_embed.proj.weight")
    pe_b = sf.get_tensor(f"{vt}.patch_embed.proj.bias")
    _check_shape("vision patch_embed proj", pe_w, (1152, 3, 16, 16))
    _check_shape("vision patch_embed bias", pe_b, (1152,))
    _add_one_tensor(writer, "vision.patch_embed.weight", pe_w, dtype)
    _add_one_tensor(writer, "vision.patch_embed.bias", pe_b, dtype)

    pe = sf.get_tensor(f"{vt}.pos_embed")
    _check_shape("vision pos_embed", pe, (1, 16385, 1152))
    # AnyRes ViT keeps a leading CLS token; the C++ runtime consumes only the
    # 16384 patch embeddings (128x128 grid at 1152 wide).  Drop the CLS row but
    # keep the 3-D layout of the reference GGUF (ne = [1152, 16384, 1]).
    _add_one_tensor(writer, "vision.pos_embed", pe[0, 1:, :].contiguous().unsqueeze(0), dtype)

    for i in range(n_vision_layers):
        for src_suf, dst_suf in VISION_BLOCK_SUFFIXES:
            t = sf.get_tensor(f"{vt}.blocks.{i}.{src_suf}")
            _add_one_tensor(writer, f"vision.blk.{i}.{dst_suf}", t, dtype)

    # merger: proj1 -> 2x2 attention pooler -> proj2
    for src, dst, shape in [
        ("proj1.weight",              "vision.merger.proj1.weight",        (2048, 1152)),
        ("proj1.bias",                "vision.merger.proj1.bias",          (2048,)),
        ("proj2.weight",              "vision.merger.proj2.weight",        (2048, 2048)),
        ("proj2.bias",                "vision.merger.proj2.bias",          (2048,)),
        ("pooler.predictor.0.weight", "vision.merger.pooler.fc0.weight",   (2048, 4096)),
        ("pooler.predictor.0.bias",   "vision.merger.pooler.fc0.bias",     (2048,)),
        ("pooler.predictor.2.weight", "vision.merger.pooler.fc2.weight",   (2048, 2048)),
        ("pooler.predictor.2.bias",   "vision.merger.pooler.fc2.bias",     (2048,)),
    ]:
        t = sf.get_tensor(f"{vis_pfx}.merger.{src}")
        _check_shape(f"vision merger {src}", t, shape)
        _add_one_tensor(writer, dst, t, dtype)


def _load_norm_stats(path: Optional[Path], max_state_dim: int) -> dict[str, np.ndarray]:
    """Load the unified norm_stats.pkl (see local/Hy-Embodied-0.5-VLA/scripts/
    compute_norm_robotwin.py) into the six ``norm.*`` GGUF tensors.

    Missing file or keys fall back to identity (zeros / ones) with a warning;
    the state statistics are padded to ``max_state_dim`` exactly like the
    released reference GGUF (tail dims are never read by the runtime).
    """
    out = {
        "state_mean":   np.zeros(max_state_dim, dtype=np.float32),
        "state_std":    np.ones (max_state_dim, dtype=np.float32),
        "action_mean":      np.zeros((20, 20), dtype=np.float32),
        "action_std":       np.ones ((20, 20), dtype=np.float32),
        "action_mean_abs":  np.zeros((20, 20), dtype=np.float32),
        "action_std_abs":   np.ones ((20, 20), dtype=np.float32),
    }
    if path is None or not path.is_file():
        print("  norm: --norm-stats not given or missing; using identity stats")
        return out
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a pickle dict, got {type(payload).__name__}")
    mapping = [
        ("qpos_mean", "state_mean"),
        ("qpos_std",  "state_std"),
        ("action_mean",    "action_mean"),
        ("action_std",     "action_std"),
        ("action_mean_abs", "action_mean_abs"),
        ("action_std_abs",  "action_std_abs"),
    ]
    for src, dst in mapping:
        if src not in payload:
            print(f"  norm: {path} missing key {src!r}; using identity for {dst}")
            continue
        arr = np.asarray(payload[src], dtype=np.float32)
        if dst.startswith("state"):
            n = min(arr.size, max_state_dim)
            out[dst][:n] = arr.reshape(-1)[:n]
        else:
            if arr.ndim != 2:
                raise SystemExit(f"{path}: {src} expected 2-D, got shape {arr.shape}")
            out[dst] = arr.astype(np.float32, copy=True)
    return out


def _write_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:
    writer.add_string  (KV("architecture"),          ARCH)
    writer.add_uint32  (KV("vlm_hidden"),            cfg["vlm_hidden"])
    writer.add_uint32  (KV("vlm_intermediate"),      cfg["vlm_intermediate"])
    writer.add_uint32  (KV("expert_hidden"),         cfg["expert_hidden"])
    writer.add_uint32  (KV("expert_intermediate"),   cfg["expert_intermediate"])
    writer.add_uint32  (KV("n_q_heads"),             cfg["n_q_heads"])
    writer.add_uint32  (KV("n_kv_heads"),            cfg["n_kv_heads"])
    writer.add_uint32  (KV("head_dim"),              cfg["head_dim"])
    writer.add_uint32  (KV("n_layers"),              cfg["n_layers"])
    writer.add_uint32  (KV("tokenizer_max_length"),  cfg["tokenizer_max_length"])
    writer.add_uint32  (KV("n_action_steps"),        cfg["n_action_steps"])
    writer.add_uint32  (KV("num_steps"),             cfg["num_steps"])
    writer.add_uint32  (KV("max_state_dim"),         cfg["max_state_dim"])
    writer.add_uint32  (KV("max_action_dim"),        cfg["max_action_dim"])
    writer.add_float32 (KV("flow_min_period"),       cfg["flow_min_period"])
    writer.add_float32 (KV("flow_max_period"),       cfg["flow_max_period"])
    writer.add_float32 (KV("rms_norm_eps"),          cfg["rms_norm_eps"])
    writer.add_float32 (KV("rope_theta"),            cfg["rope_theta"])
    writer.add_float32 (KV("rope_dynamic_alpha"),    cfg["rope_dynamic_alpha"])
    # informational metadata (not read by the runtime)
    writer.add_uint32  (KV("vocab_size"),            cfg["vocab_size"])
    writer.add_uint32  (KV("proj_width"),            cfg["proj_width"])
    writer.add_uint32  (KV("chunk_size"),            cfg["n_action_steps"])
    writer.add_bool    (KV("use_cache"),             cfg["use_cache"])
    writer.add_bool    (KV("use_video_encoder"),     cfg["use_video_encoder"])
    writer.add_bool    (KV("visual_segment_isolation"), cfg["visual_segment_isolation"])
    writer.add_string  (KV("format_stage"),          "full")
    writer.add_uint32  (KV("spacetime_layer_stride"), cfg["spacetime_layer_stride"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
        help="Hy-VLA checkpoint dir (model.safetensors + config.json)")
    ap.add_argument("--out", type=Path, default=None,
        help="Output GGUF path (default: <ckpt>/hy_vla.gguf)")
    ap.add_argument("--norm-stats", type=Path, default=None,
        help="norm_stats.pkl from compute_norm_robotwin.py (optional; identity if missing)")
    ap.add_argument("--min-period", type=float, default=0.004,
        help="Flow-matching min period (default 0.004, matches the released ckpt)")
    ap.add_argument("--max-period", type=float, default=4.0,
        help="Flow-matching max period (default 4.0, matches the released ckpt)")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16",
        help="Weight storage dtype (default bf16; norm stats are always F32)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = (args.out or ckpt / "hy_vla.gguf").resolve()
    sf_path  = ckpt / "model.safetensors"
    cfg_path = ckpt / "config.json"
    if not sf_path.is_file():
        raise SystemExit(f"missing {sf_path}")
    if not cfg_path.is_file():
        raise SystemExit(f"missing {cfg_path}")

    cfg_json = json.loads(cfg_path.read_text())
    vlm_cfg  = cfg_json.get("vlm_config_dict") or {}
    if not vlm_cfg:
        raise SystemExit(
            "config.json has no 'vlm_config_dict' (not a self-contained Hy-VLA "
            "checkpoint); the released Hy-Embodied-0.5-VLA-RoboTwin ckpt embeds it"
        )
    text_cfg = vlm_cfg.get("text_config") or vlm_cfg
    vis_cfg  = vlm_cfg.get("vision_config") or {}

    def _get(d: dict, *names: str, default=None):
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return default

    cfg = {
        "vlm_hidden":            int(_get(text_cfg, "hidden_size")),
        "vlm_intermediate":      int(_get(text_cfg, "intermediate_size")),
        "n_q_heads":             int(_get(text_cfg, "num_attention_heads")),
        "n_kv_heads":            int(_get(text_cfg, "num_key_value_heads",
                                          default=_get(text_cfg, "num_attention_heads"))),
        "head_dim":              int(_get(text_cfg, "head_dim",
                                          default=_get(text_cfg, "attention_head_dim"))),
        "n_layers":              int(_get(text_cfg, "num_hidden_layers")),
        "vocab_size":            int(_get(text_cfg, "vocab_size")),
        "rms_norm_eps":          float(_get(text_cfg, "rms_norm_eps", default=1e-5)),
        "rope_theta":            float(_get(text_cfg, "rope_theta", default=10000.0)),
        "tokenizer_max_length":  int(_get(cfg_json, "tokenizer_max_length", default=64)),
        "n_action_steps":        int(_get(cfg_json, "n_action_steps",
                                          default=_get(cfg_json, "chunk_size", default=40))),
        "num_steps":             int(_get(cfg_json, "num_steps", default=10)),
        "max_state_dim":         int(_get(cfg_json, "max_state_dim", default=32)),
        "max_action_dim":        int(_get(cfg_json, "max_action_dim", default=32)),
        "proj_width":            int(_get(cfg_json, "proj_width", default=1024)),
        "use_cache":             bool(_get(cfg_json, "use_cache", default=True)),
        "use_video_encoder":     bool(_get(cfg_json, "use_video_encoder", default=False)),
        "visual_segment_isolation": bool(_get(cfg_json, "visual_segment_isolation", default=False)),
        "spacetime_layer_stride": int(_get(cfg_json, "spacetime_layer_stride", default=4)),
        "expert_hidden":         int(_get(cfg_json, "proj_width", default=1024)),
        "expert_intermediate":   int(_get(cfg_json, "expert_intermediate", default=2048)),
        "flow_min_period":       args.min_period,
        "flow_max_period":       args.max_period,
        "rope_dynamic_alpha":    1000.0,
    }

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")
    keys = set(sf.keys())

    pfx_exp  = _pick_prefix(keys, PFX_EXP_CANDIDATES, "layers.0.input_layernorm_v.weight")
    pfx_vlm  = _pick_prefix(keys, PFX_VLM_CANDIDATES, "layers.0.input_layernorm.weight")
    pfx_head = _pick_prefix(keys, PFX_LM_HEAD_CANDIDATES, "")
    pfx_vis  = _pick_prefix(keys, PFX_VIS_CANDIDATES, "vision_tower.blocks.0.norm1.weight")
    pfx_proj = _pick_prefix(keys, PFX_PROJ_CANDIDATES, "state_proj.weight")

    n_layers = cfg["n_layers"]
    n_vision = int(_get(vis_cfg, "num_hidden_layers", default=27))
    if n_layers != 32 or n_vision != 27:
        print(f"[warn] n_layers={n_layers} vision_layers={n_vision} "
              "(expected 32 / 27 for the released Hy-Embodied-0.5-VLA)")

    # ---- hard shape checks against the released architecture ----
    qh, hdim = cfg["n_q_heads"] * cfg["head_dim"], cfg["head_dim"]
    kvdim    = cfg["n_kv_heads"] * cfg["head_dim"]
    hidden, interm = cfg["vlm_hidden"], cfg["vlm_intermediate"]
    eh, ei = cfg["expert_hidden"], cfg["expert_intermediate"]

    q0  = sf.get_tensor(f"{pfx_vlm}.layers.0.self_attn.q_proj.weight")
    _check_shape("vlm q_proj", q0, (qh, hidden))
    k0  = sf.get_tensor(f"{pfx_vlm}.layers.0.self_attn.k_proj.weight")
    _check_shape("vlm k_proj", k0, (kvdim, hidden))
    g0  = sf.get_tensor(f"{pfx_vlm}.layers.0.mlp.gate_proj.weight")
    _check_shape("vlm gate_proj", g0, (interm, hidden))

    qv0 = sf.get_tensor(f"{pfx_exp}.layers.0.self_attn.q_proj_v.weight")
    _check_shape("expert q_proj_v", qv0, (qh, eh))
    kv0 = sf.get_tensor(f"{pfx_exp}.layers.0.self_attn.k_proj_v.weight")
    _check_shape("expert k_proj_v", kv0, (kvdim, eh))
    ov0 = sf.get_tensor(f"{pfx_exp}.layers.0.self_attn.o_proj_v.weight")
    _check_shape("expert o_proj_v", ov0, (eh, qh))
    gv0 = sf.get_tensor(f"{pfx_exp}.layers.0.mlp_v.gate_proj.weight")
    _check_shape("expert gate_proj_v", gv0, (ei, eh))

    qq = sf.get_tensor(f"{pfx_vis}.vision_tower.blocks.0.attn.qkv.weight")
    _check_shape("vision qkv", qq, (3 * 1152, 1152))
    f1 = sf.get_tensor(f"{pfx_vis}.vision_tower.blocks.0.mlp.fc1.weight")
    _check_shape("vision fc1", f1, (int(_get(vis_cfg, "intermediate_size", default=4304)), 1152))

    head_w = sf.get_tensor(pfx_head)
    _check_shape("lm_head", head_w, (cfg["vocab_size"], hidden))
    if cfg["vocab_size"] != int(head_w.shape[0]):
        raise SystemExit(f"vocab mismatch: cfg={cfg['vocab_size']} ckpt={head_w.shape[0]}")

    print(f"resolved cfg: vlm={hidden}/{interm} heads={cfg['n_q_heads']}q/{cfg['n_kv_heads']}kv"
          f"x{hdim} layers={n_layers} vision={n_vision} expert={eh}/{ei} "
          f"vocab={cfg['vocab_size']} chunk={cfg['n_action_steps']} steps={cfg['num_steps']} "
          f"state={cfg['max_state_dim']} action={cfg['max_action_dim']} "
          f"flow={cfg['flow_min_period']}/{cfg['flow_max_period']} dtype={args.dtype}")

    print("loading normalization statistics...")
    stats = _load_norm_stats(args.norm_stats, cfg["max_state_dim"])

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH, use_temp_file=True)
    _write_kv(writer, cfg)

    # tied lm_head == text embedding table.  The C++ runtime fetches text
    # embeddings by row from the stable name (no HF "model." wrapper):
    # fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight").
    _add_one_tensor(writer, "dual_tower.vlm.model.language_model.lm_head.weight", head_w, args.dtype)

    # VLM text tower: 11 non-_v + 9 _v per layer
    _stream_layers(writer, sf, pfx_vlm, "vlm", n_layers, TEXT_LAYER_SUFFIXES, args.dtype)
    _stream_layers(writer, sf, pfx_vlm, "vlm", n_layers, V_LAYER_SUFFIXES, args.dtype)
    _add_one_tensor(writer, "vlm.output_norm.weight",
                    sf.get_tensor(f"{pfx_vlm}.norm.weight"), args.dtype)

    # suffix expert: 9 _v per layer only (C++ loads no non-_v expert tensors)
    _stream_layers(writer, sf, pfx_exp, "expert", n_layers, V_LAYER_SUFFIXES, args.dtype)
    _add_one_tensor(writer, "expert.output_norm.weight",
                    sf.get_tensor(f"{pfx_exp}.norm.weight"), args.dtype)

    # anyres vision tower + merger
    _stream_vision(writer, sf, pfx_vis, n_vision, args.dtype)

    for suf in ["state_proj.weight", "state_proj.bias",
                "action_in_proj.weight", "action_in_proj.bias",
                "action_time_mlp_in.weight", "action_time_mlp_in.bias",
                "action_time_mlp_out.weight", "action_time_mlp_out.bias",
                "action_out_proj.weight", "action_out_proj.bias"]:
        _add_one_tensor(writer, suf, sf.get_tensor(_join_key(pfx_proj, suf)), args.dtype)

    for k, v in stats.items():
        writer.add_tensor(f"norm.{k}", v, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    n_tensors = sum(len(t) for t in writer.tensors)
    print(f"done. {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB, "
          f"{n_tensors} tensors, weights in {args.dtype})")
    print("note: run with VLA_HY_VLA_TEXT_LAYERS=32 VLA_HY_VLA_VISION_LAYERS=27 "
          "to load the full VLM + vision tower on Orin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
