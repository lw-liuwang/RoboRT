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

"""Convert a FasterWAM .pt checkpoint into the single-GGUF format consumed by
the RoboRT ``fasterwam`` backend (``src/models/fasterwam/fasterwam.cpp``).

FasterWAM (hustvl/FasterWAM) consists of a Wan2.2-TI2V-5B video DiT and a
SparseActionDiT action expert joined by a Mixture-of-Transformers (MoT) with
sparse cross-attention at selected condition layers.  At inference time the
video DiT runs once (``prefill_video_cache``) and the action expert denoises
over N steps using the cached video KV.

The .pt checkpoint stores weights under the ``"mot"`` top-level key:
  mot.mixtures.video.*       -- WanVideoDiT 30 layers (hidden=3072)
  mot.mixtures.action.*      -- SparseActionDiT 30 layers (hidden=1024)
  mot.video_kv_fusion_logits -- KV fusion logits (8 condition layers)
  proprio_encoder.*          -- Linear(proprio_dim -> text_dim=4096)

Checkpoint GGUF tensor naming (C++ names fixed in fasterwam.cpp):

  Video expert:
    video.patch_embed.{weight,bias}           patch embedding conv
    video.time_embed.{0,2}.{weight,bias}      time MLP (2 linears)
    video.time_proj.1.{weight,bias}           time projection
    video.text_embed.{0,2}.{weight,bias}      text MLP (2 linears)
    video.head.{weight,bias}                  output head linear
    video.head_mod                            head modulation [1,2,h]
    video.blk.{i}.self_attn.{q,k,v,o}.{weight,bias}   self-attn
    video.blk.{i}.self_attn.norm_{q,k}.weight          QK norm
    video.blk.{i}.ffn.{0,2}.{weight,bias}              FFN (2 linears)
    video.blk.{i}.modulation                  [1,6,h] DiT modulation
    video.blk.{i}.norm3.{weight,bias}         cross-attn pre-norm
    video.blk.{i}.cross_attn.{q,k,v,o}.{weight,bias}  cross-attn
    video.blk.{i}.cross_attn.norm_{q,k}.weight         cross QK norm

  Action expert (condition layers have cross_attn, others do not):
    action.action_encoder.{weight,bias}        action input proj
    action.head.{weight,bias}                  action output proj
    action.time_embed.{0,2}.{weight,bias}      time MLP
    action.time_proj.1.{weight,bias}           time projection
    action.text_embed.{0,2}.{weight,bias}      text MLP
    action.blk.{i}.self_attn.{q,k,v,o}.{weight,bias}
    action.blk.{i}.self_attn.norm_{q,k}.weight
    action.blk.{i}.ffn.{0,2}.{weight,bias}
    action.blk.{i}.modulation
    action.blk.{i}.norm3.{weight,bias}
    action.blk.{i}.cross_attn.{q,k,v,o}.{weight,bias}  (condition layers only)
    action.blk.{i}.cross_attn.norm_{q,k}.weight         (condition layers only)

  KV fusion logits (condition layers, sparse):
    video_kv_fusion_logits.{j}                [1] or [4] scalar logits

  Proprio encoder:
    proprio_encoder.{weight,bias}              Linear(proprio_dim -> text_dim)

  Normalization statistics (always F32):
    norm.state_mean   [state_dim]
    norm.state_std    [state_dim]
    norm.action_mean  [action_horizon, action_dim]
    norm.action_std   [action_horizon, action_dim]

Usage
-----
python src/models/fasterwam/convert_fasterwam_to_gguf.py \\
    --ckpt /path/to/models/FasterWAM/robotwin/step_029355.pt \\
    --dataset-stats /path/to/models/FasterWAM/robotwin/dataset_stats.json \\
    --out models/FasterWAM/robotwin/fasterwam_robotwin.gguf

All weights are stored in BF16 by default; pass ``--dtype f32`` for F32.
Normalization statistics are always stored as F32.

Architecture config defaults match the released hustvl/FasterWAM checkpoint:
  video: 30 layers, hidden=3072, ffn=14336, heads=24, head_dim=128
  action: 30 layers, hidden=1024, ffn=4096, heads=24 (cond) / 8 (non-cond), head_dim=128
  action_dim=14, text_dim=4096, freq_dim=256
  condition_layers=[0,4,8,12,16,20,24,28]
  infer_shift=5.0, num_train_timesteps=1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "third_party" / "gguf-py"))
import gguf

ARCH = "fasterwam"
KV   = lambda name: f"{ARCH}.{name}"


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    """Return raw BF16 bits as a uint16 numpy array (portable across torch versions)."""
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


def _add_tensor(writer: gguf.GGUFWriter, name: str, t: torch.Tensor, dtype: str) -> None:
    if dtype == "f32":
        writer.add_tensor(name, _f32_np(t), raw_dtype=gguf.GGMLQuantizationType.F32)
    else:  # bf16
        writer.add_tensor(name, _bf16_to_u16_bytes(t),
                          raw_shape=list(t.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)


def _add_f32(writer: gguf.GGUFWriter, name: str, arr: np.ndarray) -> None:
    writer.add_tensor(name, arr.astype(np.float32, copy=False),
                      raw_dtype=gguf.GGMLQuantizationType.F32)


# ---------------------------------------------------------------------------
# KV metadata
# ---------------------------------------------------------------------------

def _write_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:
    writer.add_string (KV("architecture"),          ARCH)
    # Video expert
    writer.add_uint32 (KV("video_hidden"),          cfg["video_hidden"])
    writer.add_uint32 (KV("video_ffn_dim"),         cfg["video_ffn_dim"])
    writer.add_uint32 (KV("video_num_heads"),       cfg["video_num_heads"])
    writer.add_uint32 (KV("video_head_dim"),        cfg["video_head_dim"])
    writer.add_uint32 (KV("video_num_layers"),      cfg["video_num_layers"])
    writer.add_uint32 (KV("video_in_dim"),          cfg["video_in_dim"])
    writer.add_uint32 (KV("video_freq_dim"),        cfg["video_freq_dim"])
    writer.add_uint32 (KV("video_text_dim"),        cfg["video_text_dim"])
    # Action expert
    writer.add_uint32 (KV("action_hidden"),         cfg["action_hidden"])
    writer.add_uint32 (KV("action_ffn_dim"),        cfg["action_ffn_dim"])
    writer.add_uint32 (KV("action_num_heads"),      cfg["action_num_heads"])
    writer.add_uint32 (KV("action_noncond_heads"),  cfg["action_noncond_heads"])
    writer.add_uint32 (KV("action_head_dim"),       cfg["action_head_dim"])
    writer.add_uint32 (KV("action_num_layers"),     cfg["action_num_layers"])
    writer.add_uint32 (KV("action_dim"),            cfg["action_dim"])
    writer.add_uint32 (KV("action_freq_dim"),       cfg["action_freq_dim"])
    writer.add_uint32 (KV("action_text_dim"),       cfg["action_text_dim"])
    # Condition layers list
    writer.add_uint32 (KV("num_condition_layers"),  len(cfg["condition_layers"]))
    for j, cl in enumerate(cfg["condition_layers"]):
        writer.add_uint32(KV(f"condition_layer_{j}"), cl)
    # Proprio / state
    writer.add_uint32 (KV("proprio_dim"),           cfg["proprio_dim"])
    writer.add_uint32 (KV("state_dim"),             cfg["state_dim"])
    # Inference config
    writer.add_uint32 (KV("action_horizon"),        cfg["action_horizon"])
    writer.add_uint32 (KV("num_inference_steps"),   cfg["num_inference_steps"])
    writer.add_uint32 (KV("num_video_frames"),      cfg["num_video_frames"])
    writer.add_float32(KV("infer_shift"),           cfg["infer_shift"])
    writer.add_uint32 (KV("num_train_timesteps"),   cfg["num_train_timesteps"])
    # Normalization
    writer.add_string (KV("norm_mode"),             cfg["norm_mode"])


# ---------------------------------------------------------------------------
# Block-level tensor writers
# ---------------------------------------------------------------------------

_ATTN_SUFFIXES = [
    ("self_attn.q.weight",       "self_attn.q.weight"),
    ("self_attn.q.bias",         "self_attn.q.bias"),
    ("self_attn.k.weight",       "self_attn.k.weight"),
    ("self_attn.k.bias",         "self_attn.k.bias"),
    ("self_attn.v.weight",       "self_attn.v.weight"),
    ("self_attn.v.bias",         "self_attn.v.bias"),
    ("self_attn.o.weight",       "self_attn.o.weight"),
    ("self_attn.o.bias",         "self_attn.o.bias"),
    ("self_attn.norm_q.weight",  "self_attn.norm_q.weight"),
    ("self_attn.norm_k.weight",  "self_attn.norm_k.weight"),
    ("ffn.0.weight",             "ffn.0.weight"),
    ("ffn.0.bias",               "ffn.0.bias"),
    ("ffn.2.weight",             "ffn.2.weight"),
    ("ffn.2.bias",               "ffn.2.bias"),
    ("modulation",               "modulation"),
    ("norm3.weight",             "norm3.weight"),
    ("norm3.bias",               "norm3.bias"),
]

_CROSS_ATTN_SUFFIXES = [
    ("cross_attn.q.weight",      "cross_attn.q.weight"),
    ("cross_attn.q.bias",        "cross_attn.q.bias"),
    ("cross_attn.k.weight",      "cross_attn.k.weight"),
    ("cross_attn.k.bias",        "cross_attn.k.bias"),
    ("cross_attn.v.weight",      "cross_attn.v.weight"),
    ("cross_attn.v.bias",        "cross_attn.v.bias"),
    ("cross_attn.o.weight",      "cross_attn.o.weight"),
    ("cross_attn.o.bias",        "cross_attn.o.bias"),
    ("cross_attn.norm_q.weight", "cross_attn.norm_q.weight"),
    ("cross_attn.norm_k.weight", "cross_attn.norm_k.weight"),
]


def _stream_blocks(writer: gguf.GGUFWriter, mot: dict[str, torch.Tensor],
                   src_prefix: str, dst_prefix: str,
                   n_layers: int, condition_layers: set[int], dtype: str) -> None:
    for i in range(n_layers):
        src_blk = f"{src_prefix}.{i}"
        dst_blk = f"{dst_prefix}.blk.{i}"

        # self-attention + FFN + modulation (all layers)
        for src_suf, dst_suf in _ATTN_SUFFIXES:
            key = f"{src_blk}.{src_suf}"
            if key not in mot:
                # non-condition action blocks lack norm3 and cross_attn; skip
                if "norm3" in src_suf:
                    continue
                raise KeyError(f"missing expected tensor {key!r} in checkpoint")
            _add_tensor(writer, f"{dst_blk}.{dst_suf}", mot[key], dtype)

        # cross-attention (condition layers only)
        if i in condition_layers:
            for src_suf, dst_suf in _CROSS_ATTN_SUFFIXES:
                key = f"{src_blk}.{src_suf}"
                _add_tensor(writer, f"{dst_blk}.{dst_suf}", mot[key], dtype)


def _stream_video_globals(writer: gguf.GGUFWriter, mot: dict[str, torch.Tensor],
                           dtype: str) -> None:
    """Write video expert global (non-block) tensors."""
    # patch_embedding.weight is a 5-D Conv3d tensor [out, in, t, h, w].
    # GGML only supports up to GGML_MAX_DIMS=4.  Flatten the spatial dims to 1
    # to get a 2-D [out, in*t*h*w] matrix that gguf can store safely.
    pe_w = mot["mixtures.video.patch_embedding.weight"]  # [3072, 48, 1, 2, 2]
    pe_w_flat = pe_w.reshape(pe_w.shape[0], -1)           # [3072, 192]
    _add_tensor(writer, "video.patch_embed.weight", pe_w_flat, dtype)
    for src, dst in [
        ("mixtures.video.patch_embedding.bias",    "video.patch_embed.bias"),
        ("mixtures.video.time_embedding.0.weight", "video.time_embed.0.weight"),
        ("mixtures.video.time_embedding.0.bias",   "video.time_embed.0.bias"),
        ("mixtures.video.time_embedding.2.weight", "video.time_embed.2.weight"),
        ("mixtures.video.time_embedding.2.bias",   "video.time_embed.2.bias"),
        ("mixtures.video.time_projection.1.weight","video.time_proj.1.weight"),
        ("mixtures.video.time_projection.1.bias",  "video.time_proj.1.bias"),
        ("mixtures.video.text_embedding.0.weight", "video.text_embed.0.weight"),
        ("mixtures.video.text_embedding.0.bias",   "video.text_embed.0.bias"),
        ("mixtures.video.text_embedding.2.weight", "video.text_embed.2.weight"),
        ("mixtures.video.text_embedding.2.bias",   "video.text_embed.2.bias"),
        ("mixtures.video.head.head.weight",        "video.head.weight"),
        ("mixtures.video.head.head.bias",          "video.head.bias"),
        ("mixtures.video.head.modulation",         "video.head_mod"),
    ]:
        _add_tensor(writer, dst, mot[src], dtype)


def _stream_action_globals(writer: gguf.GGUFWriter, mot: dict[str, torch.Tensor],
                            proprio: dict[str, torch.Tensor], dtype: str) -> None:
    """Write action expert global (non-block) tensors + proprio encoder."""
    for src, dst in [
        ("mixtures.action.action_encoder.weight",  "action.action_encoder.weight"),
        ("mixtures.action.action_encoder.bias",    "action.action_encoder.bias"),
        ("mixtures.action.head.weight",            "action.head.weight"),
        ("mixtures.action.head.bias",              "action.head.bias"),
        ("mixtures.action.time_embedding.0.weight","action.time_embed.0.weight"),
        ("mixtures.action.time_embedding.0.bias",  "action.time_embed.0.bias"),
        ("mixtures.action.time_embedding.2.weight","action.time_embed.2.weight"),
        ("mixtures.action.time_embedding.2.bias",  "action.time_embed.2.bias"),
        ("mixtures.action.time_projection.1.weight","action.time_proj.1.weight"),
        ("mixtures.action.time_projection.1.bias", "action.time_proj.1.bias"),
        ("mixtures.action.text_embedding.0.weight","action.text_embed.0.weight"),
        ("mixtures.action.text_embedding.0.bias",  "action.text_embed.0.bias"),
        ("mixtures.action.text_embedding.2.weight","action.text_embed.2.weight"),
        ("mixtures.action.text_embedding.2.bias",  "action.text_embed.2.bias"),
    ]:
        _add_tensor(writer, dst, mot[src], dtype)

    # proprio encoder: Linear(proprio_dim -> text_dim)
    _add_tensor(writer, "proprio_encoder.weight", proprio["weight"], dtype)
    _add_tensor(writer, "proprio_encoder.bias",   proprio["bias"],   dtype)


def _stream_kv_fusion(writer: gguf.GGUFWriter, mot: dict[str, torch.Tensor],
                       dtype: str) -> None:
    """Write video_kv_fusion_logits (one tensor per condition layer)."""
    j = 0
    while f"video_kv_fusion_logits.{j}" in mot:
        _add_tensor(writer, f"video_kv_fusion_logits.{j}",
                    mot[f"video_kv_fusion_logits.{j}"], dtype)
        j += 1
    if j == 0:
        print("  [warn] no video_kv_fusion_logits found in checkpoint")


# ---------------------------------------------------------------------------
# Normalization statistics loader
# ---------------------------------------------------------------------------

def _load_norm_stats(stats_path: Path | None, state_dim: int, action_horizon: int,
                     action_dim: int, norm_mode: str) -> dict[str, np.ndarray]:
    """Load dataset_stats.json normalization arrays.

    Returns a dict with keys: state_mean, state_std, action_mean, action_std.
    Falls back to identity (zeros/ones) if file is missing or keys are absent.
    """
    out = {
        "state_mean":  np.zeros(state_dim,                       dtype=np.float32),
        "state_std":   np.ones (state_dim,                       dtype=np.float32),
        "action_mean": np.zeros((action_horizon, action_dim),    dtype=np.float32),
        "action_std":  np.ones ((action_horizon, action_dim),    dtype=np.float32),
    }

    if stats_path is None or not Path(stats_path).is_file():
        print("  norm: --dataset-stats not given or file missing; using identity stats")
        return out

    with open(stats_path) as f:
        stats = json.load(f)

    mode_key = {"mean_std": ("stepwise_mean", "stepwise_std"),
                "global_mean_std": ("global_mean", "global_std"),
                "quantile": ("stepwise_q01", "stepwise_q99"),
                "global_quantile": ("global_q01", "global_q99")}.get(norm_mode.lower())
    if mode_key is None:
        print(f"  norm: unknown norm_mode {norm_mode!r}; defaulting to stepwise_mean/stepwise_std")
        mode_key = ("stepwise_mean", "stepwise_std")

    mean_key, std_key = mode_key

    # state
    state_stats = stats.get("state", {}).get("default", {})
    if mean_key in state_stats and std_key in state_stats:
        mean = np.array(state_stats[mean_key], dtype=np.float64).reshape(-1)[:state_dim]
        std  = np.array(state_stats[std_key],  dtype=np.float64).reshape(-1)[:state_dim]
        n = len(mean)
        out["state_mean"][:n] = mean.astype(np.float32)
        out["state_std"][:n]  = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        print(f"  norm: loaded state {mean_key}/{std_key} shape=({n},)")
    else:
        print(f"  norm: state stats key {mean_key!r} missing; using identity")

    # action
    action_stats = stats.get("action", {}).get("default", {})
    if mean_key in action_stats and std_key in action_stats:
        mean = np.array(action_stats[mean_key], dtype=np.float64)
        std  = np.array(action_stats[std_key],  dtype=np.float64)
        # may be (H, D) for stepwise or (D,) for global
        if mean.ndim == 1:
            mean = np.tile(mean[np.newaxis, :], (action_horizon, 1))
            std  = np.tile(std [np.newaxis, :], (action_horizon, 1))
        h = min(mean.shape[0], action_horizon)
        d = min(mean.shape[1], action_dim)
        out["action_mean"][:h, :d] = mean[:h, :d].astype(np.float32)
        out["action_std"][:h,  :d] = np.where(std[:h, :d] < 1e-8, 1.0,
                                               std[:h, :d]).astype(np.float32)
        print(f"  norm: loaded action {mean_key}/{std_key} shape=({h},{d})")
    else:
        print(f"  norm: action stats key {mean_key!r} missing; using identity")

    return out


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def _check(name: str, t: torch.Tensor, expected: tuple) -> None:
    if tuple(t.shape) != tuple(expected):
        raise SystemExit(
            f"shape mismatch for {name}: got {tuple(t.shape)}, expected {tuple(expected)}\n"
            "checkpoint does not match the expected FasterWAM architecture"
        )


def _validate_shapes(mot: dict, proprio: dict, cfg: dict) -> None:
    vh, vn = cfg["video_hidden"], cfg["video_num_heads"]
    ah, an = cfg["action_hidden"], cfg["action_num_heads"]
    hd = cfg["video_head_dim"]
    vd = cfg["action_dim"]
    td = cfg["video_text_dim"]

    # Video block 0
    _check("video q_proj",  mot["mixtures.video.blocks.0.self_attn.q.weight"],  (vn*hd, vh))
    _check("video ffn.0",   mot["mixtures.video.blocks.0.ffn.0.weight"],        (cfg["video_ffn_dim"], vh))

    # Action block 0 (condition layer)
    _check("action q_proj", mot["mixtures.action.blocks.0.self_attn.q.weight"], (vn*hd, ah))
    _check("action ffn.0",  mot["mixtures.action.blocks.0.ffn.0.weight"],       (cfg["action_ffn_dim"], ah))
    _check("action_encoder",mot["mixtures.action.action_encoder.weight"],       (ah, vd))

    # Proprio encoder
    _check("proprio weight", proprio["weight"], (td, cfg["proprio_dim"]))
    _check("proprio bias",   proprio["bias"],   (td,))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
        help="FasterWAM .pt checkpoint file (step_XXXXXX.pt with 'mot' + 'proprio_encoder' keys)")
    ap.add_argument("--out", type=Path, default=None,
        help="Output GGUF path (default: <ckpt_dir>/fasterwam.gguf)")
    ap.add_argument("--dataset-stats", type=Path, default=None,
        help="dataset_stats.json (from hustvl/FasterWAM/robotwin/ or /libero/)")
    ap.add_argument("--norm-mode", default="mean_std",
        choices=["mean_std", "global_mean_std", "quantile", "global_quantile"],
        help="Which statistics to use for normalization (default: mean_std = stepwise)")
    ap.add_argument("--action-dim", type=int, default=14,
        help="Robot action dimension (default: 14 for RoboTwin dual-arm)")
    ap.add_argument("--proprio-dim", type=int, default=14,
        help="Robot state/proprioception dimension (default: 14 for RoboTwin)")
    ap.add_argument("--action-horizon", type=int, default=32,
        help="Action chunk length (default: 32)")
    ap.add_argument("--num-video-frames", type=int, default=9,
        help="Video latent frames = (T-1)//4+1 (default: 9 for T=33)")
    ap.add_argument("--num-inference-steps", type=int, default=20,
        help="Action denoising steps at inference (default: 20)")
    ap.add_argument("--infer-shift", type=float, default=5.0,
        help="Flow-matching sigma shift for inference schedule (default: 5.0)")
    ap.add_argument("--num-train-timesteps", type=int, default=1000,
        help="Number of training diffusion timesteps (default: 1000)")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16",
        help="Weight storage dtype (default bf16; norm stats always F32)")
    args = ap.parse_args()

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    out = (args.out or ckpt_path.parent / "fasterwam.gguf").resolve()

    # ---- load checkpoint ----
    print(f"loading {ckpt_path}  (this may take ~1 min for 10 GiB checkpoint)")
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "mot" not in raw:
        raise SystemExit(f"unexpected checkpoint format: top-level keys = {list(raw.keys())}")

    mot     = raw["mot"]          # all MoT weights
    proprio = raw["proprio_encoder"]

    # ---- detect architecture from checkpoint ----
    # Count blocks
    video_layers  = max(int(k.split(".")[3]) for k in mot if k.startswith("mixtures.video.blocks.")) + 1
    action_layers = max(int(k.split(".")[3]) for k in mot if k.startswith("mixtures.action.blocks.")) + 1
    if video_layers != action_layers:
        raise SystemExit(f"video ({video_layers}) and action ({action_layers}) layer counts differ")

    # Condition layers = action blocks that have cross_attn
    condition_layers = sorted({
        int(k.split(".")[3])
        for k in mot
        if k.startswith("mixtures.action.blocks.") and ".cross_attn." in k
    })
    if not condition_layers:
        raise SystemExit("no condition (cross-attn) layers found in checkpoint")

    # Infer shapes from checkpoint tensors.
    # video self-attn q: [num_heads*head_dim, hidden]  -> q_out = num_heads * head_dim = hidden (Wan: full dim)
    # For Wan2.2: num_heads=24, head_dim=128, hidden=3072 => q_out=3072=hidden
    # Infer head_dim from config yaml default (128), then derive num_heads from q_out/head_dim
    # We detect head_dim from num_heads: time_proj.1 output is 6*hidden, so heads can't be inferred
    # from it. Instead use the canonical Wan2.2 head_dim=128 (validated by shape consistency below).
    HEAD_DIM_WAN22 = 128  # canonical for Wan2.2-TI2V-5B
    video_hidden   = int(mot["mixtures.video.blocks.0.self_attn.q.weight"].shape[1])
    action_hidden  = int(mot["mixtures.action.blocks.0.self_attn.q.weight"].shape[1])
    video_ffn_w    = mot["mixtures.video.blocks.0.ffn.0.weight"].shape
    action_ffn_w   = mot["mixtures.action.blocks.0.ffn.0.weight"].shape
    video_time     = mot["mixtures.video.time_embedding.0.weight"].shape
    action_text    = mot["mixtures.action.text_embedding.0.weight"].shape

    head_dim          = HEAD_DIM_WAN22
    num_heads_video   = video_hidden  // head_dim   # 3072 / 128 = 24
    num_heads_action  = int(mot["mixtures.action.blocks.0.self_attn.q.weight"].shape[0]) // head_dim  # 3072/128=24
    ncl1              = int(mot["mixtures.action.blocks.1.self_attn.q.weight"].shape[0])
    noncond_num_heads = ncl1 // head_dim             # 1024/128=8 (noncond blocks have narrower q)

    cfg = {
        # Video expert
        "video_hidden":    video_hidden,
        "video_ffn_dim":   int(video_ffn_w[0]),
        "video_num_heads": num_heads_video,
        "video_head_dim":  head_dim,
        "video_num_layers": video_layers,
        "video_in_dim":    int(mot["mixtures.video.patch_embedding.weight"].shape[1]),  # VAE z_dim channels
        "video_freq_dim":  int(video_time[1]),       # sinusoidal freq dim
        "video_text_dim":  int(mot["mixtures.video.text_embedding.0.weight"].shape[1]),
        # Action expert
        "action_hidden":       action_hidden,
        "action_ffn_dim":      int(action_ffn_w[0]),
        "action_num_heads":    num_heads_action,
        "action_noncond_heads": noncond_num_heads,
        "action_head_dim":     head_dim,
        "action_num_layers":   action_layers,
        "action_dim":          args.action_dim,
        "action_freq_dim":     int(mot["mixtures.action.time_embedding.0.weight"].shape[1]),
        "action_text_dim":     int(action_text[1]),
        # MoT
        "condition_layers":    condition_layers,
        # Proprio / state
        "proprio_dim":    int(proprio["weight"].shape[1]),
        "state_dim":      args.proprio_dim,
        # Inference
        "action_horizon":       args.action_horizon,
        "num_inference_steps":  args.num_inference_steps,
        "num_video_frames":     args.num_video_frames,
        "infer_shift":          args.infer_shift,
        "num_train_timesteps":  args.num_train_timesteps,
        "norm_mode":            args.norm_mode,
    }

    print(f"detected cfg:")
    print(f"  video: {video_layers}L hidden={video_hidden} ffn={cfg['video_ffn_dim']}"
          f" heads={num_heads_video} head_dim={head_dim} in_dim={cfg['video_in_dim']}"
          f" text_dim={cfg['video_text_dim']}")
    print(f"  action: {action_layers}L hidden={action_hidden} ffn={cfg['action_ffn_dim']}"
          f" heads={num_heads_action}(cond)/{noncond_num_heads}(noncond)"
          f" action_dim={cfg['action_dim']}")
    print(f"  condition_layers: {condition_layers}")
    print(f"  proprio_dim={cfg['proprio_dim']} state_dim={cfg['state_dim']}")
    print(f"  action_horizon={cfg['action_horizon']} num_inference_steps={cfg['num_inference_steps']}"
          f" num_video_frames={cfg['num_video_frames']}")

    _validate_shapes(mot, proprio, cfg)

    # ---- normalization stats ----
    print("loading normalization statistics...")
    norm_stats = _load_norm_stats(
        args.dataset_stats,
        state_dim=cfg["state_dim"],
        action_horizon=cfg["action_horizon"],
        action_dim=cfg["action_dim"],
        norm_mode=args.norm_mode,
    )

    # ---- write GGUF ----
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}  (dtype={args.dtype})")
    # use_temp_file=False: stream tensors directly to the output file to avoid
    # /tmp space exhaustion (the ~10 GiB model overflows the SpooledTemporaryFile
    # max_size=256 MiB and spills to /tmp which may be small).
    writer = gguf.GGUFWriter(str(out), arch=ARCH, use_temp_file=False)
    _write_kv(writer, cfg)

    # Global tensors: video expert
    print("  writing video global tensors...")
    _stream_video_globals(writer, mot, args.dtype)

    # Video blocks
    print(f"  writing {video_layers} video blocks...")
    condition_layer_set = set(condition_layers)
    _stream_blocks(writer, mot,
                   src_prefix="mixtures.video.blocks",
                   dst_prefix="video",
                   n_layers=video_layers,
                   condition_layers=set(range(video_layers)),  # all video blocks have cross_attn
                   dtype=args.dtype)

    # Global tensors: action expert + proprio
    print("  writing action global tensors + proprio encoder...")
    _stream_action_globals(writer, mot, proprio, args.dtype)

    # Action blocks
    print(f"  writing {action_layers} action blocks "
          f"({len(condition_layers)} condition, {action_layers - len(condition_layers)} non-condition)...")
    _stream_blocks(writer, mot,
                   src_prefix="mixtures.action.blocks",
                   dst_prefix="action",
                   n_layers=action_layers,
                   condition_layers=condition_layer_set,
                   dtype=args.dtype)

    # KV fusion logits
    print("  writing video_kv_fusion_logits...")
    _stream_kv_fusion(writer, mot, args.dtype)

    # Normalization statistics (always F32)
    print("  writing normalization statistics (F32)...")
    for k, v in norm_stats.items():
        _add_f32(writer, f"norm.{k}", v)

    print("flushing to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mib = out.stat().st_size / (1024 * 1024)
    n_tensors = sum(len(t) for t in writer.tensors)
    print(f"done. {out}")
    print(f"  {size_mib:.1f} MiB  |  {n_tensors} tensors  |  weights in {args.dtype}")
    print()
    print("next steps:")
    print("  1. implement fasterwam.cpp inference (src/models/fasterwam/fasterwam.cpp)")
    print("  2. rebuild:  cmake --build build --target robort-server")
    print(f"  3. serve:    robort-server {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
