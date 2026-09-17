#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Export the Qwen3-VL vision tower of lingbot-vla-v2-6b (Robbyant) as a
# llama.cpp mtmd mmproj GGUF with projector_type "qwen3vl_merger".
#
# The RoboRT mtmd build (third_party/mtmd) ships clip_graph_qwen3vl, which
# implements the full Qwen3-VL ViT: temporal-sliced conv patch embed,
# learnable absolute pos-embedding table (bicubic/bilinear resize), 3D M-RoPE
# vision attention, GELU (tanh) MLPs, 2x2 spatial merger, and DeepStack
# feature extraction at layers {5, 11, 17} (Qwen3-VL-4B-Instruct default).
# DeepStack features are returned concatenated to the main embeddings along
# the feature dim: [2560 * (1 + 3), n_tokens].
#
# Usage:
#   python convert_lingbot_mmproj_to_gguf.py \
#     --ckpt /path/to/lingbot-vla-v2-6b --out lingbot-mmproj-f32.gguf [--dtype bf16]

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "third_party" / "gguf-py"))
import gguf

# lingbot-vla-v2-6b checkpoint prefixes
PFX = "model.qwenvl_with_expert.qwenvl.model"
VIS = f"{PFX}.visual"

# Qwen3-VL-4B-Instruct vision config (tokenizer_path of the lingbot robotwin config)
N_LAYER = 24
N_HEAD = 16
N_EMBD = 1024
N_FF = 4096
PATCH = 16
N_POS_EMBD = 2304  # 48 * 48
PROJ_DIM = 2560  # out_hidden_size
DEEPSTACK_IDX = [5, 11, 17]
LN_EPS = 1e-6


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).contiguous().cpu().numpy()


def _hf_interpolate_pos_embed(table: torch.Tensor, n_grid: int, h: int, w: int) -> torch.Tensor:
    """Exact port of HF Qwen3VLVisionModel.fast_pos_embed_interpolate (transformers
    4.57.x): plain 4-corner bilinear, align_corners=True (linspace endpoints),
    NO antialias, computed in fp32.

    mtmd's runtime resize_position_embeddings() uses BILINEAR|ANTIALIAS, which
    differs from HF when downscaling (48x48 -> 16x16 here). Pre-interpolating
    the table to the fixed lingbot image grid makes the runtime resize a no-op
    (resize is skipped when height == n_per_side), guaranteeing HF parity.

    table: [n_grid*n_grid, D] raster (row-major) order. Returns [h*w, D].
    """
    table = table.to(torch.float32)
    h_idxs = torch.linspace(0, n_grid - 1, h)
    w_idxs = torch.linspace(0, n_grid - 1, w)
    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_idxs.int() + 1).clamp(max=n_grid - 1)
    w_ceil = (w_idxs.int() + 1).clamp(max=n_grid - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    base_h = h_floor * n_grid
    base_h_ceil = h_ceil * n_grid

    idx = [
        (base_h[:, None] + w_floor[None, :]).flatten(),
        (base_h[:, None] + w_ceil[None, :]).flatten(),
        (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
        (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
    ]
    wt = [
        ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
        ((1 - dh)[:, None] * dw[None, :]).flatten(),
        (dh[:, None] * (1 - dw)[None, :]).flatten(),
        (dh[:, None] * dw[None, :]).flatten(),
    ]
    out = None
    for i, x in enumerate(idx):
        contrib = table[x] * wt[i][:, None]
        out = contrib if out is None else out + contrib
    return out


class ShardReader:
    def __init__(self, ckpt: Path):
        idx_path = ckpt / "model.safetensors.index.json"
        single = ckpt / "model.safetensors"
        self.handles = {}
        if idx_path.is_file():
            import json
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
        if self.weight_map is not None:
            shard = self.weight_map[name]
        else:
            shard = self.shards[0]
        return self.handles[shard].get_tensor(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export lingbot-vla-v2 Qwen3-VL vision tower as mtmd mmproj GGUF")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="lingbot-vla-v2-6b checkpoint directory (model-0000X-of-00006.safetensors)")
    ap.add_argument("--out", type=Path, required=True, help="Output mmproj GGUF path")
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32",
                    help="Weight dtype for linear/conv weights (norms and biases are always F32). Default f32.")
    ap.add_argument("--img-size", type=int, default=256,
                    help="Fixed square image size the model will always receive (default 256). "
                         "The pos-embed table is pre-interpolated to this grid so the runtime "
                         "resize (which uses antialias bilinear, unlike HF) is skipped.")
    ap.add_argument("--raw-pos-embed", action="store_true",
                    help="Store the original 48x48 pos-embed table and let mtmd resize at runtime "
                         "(NOT bit-exact vs HF: mtmd uses antialias bilinear).")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    rd = ShardReader(ckpt)
    rd.open(ckpt)
    keys = rd.keys()

    def get(name: str) -> torch.Tensor:
        if name not in keys:
            raise SystemExit(f"missing tensor {name}")
        return rd.get(name)

    def add(name: str, t: torch.Tensor, force_f32: bool = False):
        if args.dtype == "bf16" and not force_f32:
            writer.add_tensor(name, _bf16_to_u16_bytes(t),
                              raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, _f32_np(t), raw_dtype=gguf.GGMLQuantizationType.F32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch="clip")
    writer.add_string("general.name", "lingbot-vla-v2-6b")
    writer.add_string("general.type", "mmproj")
    writer.add_string("general.finetune", "lingbot-vla-v2")
    writer.add_string("general.size_label", "0.4B")
    writer.add_uint32("general.quantization_version", 2)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    writer.add_bool("clip.use_gelu", True)
    img_grid = args.img_size // PATCH  # patches per side
    writer.add_uint32("clip.vision.image_size", args.img_size if not args.raw_pos_embed
                      else int(N_POS_EMBD ** 0.5 * PATCH))
    writer.add_uint32("clip.vision.patch_size", PATCH)
    writer.add_uint32("clip.vision.embedding_length", N_EMBD)
    writer.add_uint32("clip.vision.feed_forward_length", N_FF)
    writer.add_uint32("clip.vision.block_count", N_LAYER)
    writer.add_uint32("clip.vision.attention.head_count", N_HEAD)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", LN_EPS)
    writer.add_uint32("clip.vision.projection_dim", PROJ_DIM)
    writer.add_uint32("clip.vision.spatial_merge_size", 2)
    writer.add_array("clip.vision.is_deepstack_layers", [i in DEEPSTACK_IDX for i in range(N_LAYER)])
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_file_type(gguf.LlamaFileType.ALL_F32 if args.dtype == "f32" else gguf.LlamaFileType.MOSTLY_BF16)

    n_tensors = 0

    # ---- patch embed: Conv3d [1024, 3, 2, 16, 16] -> two temporal conv2d slices ----
    w = get(f"{VIS}.patch_embed.proj.weight")
    assert tuple(w.shape) == (N_EMBD, 3, 2, PATCH, PATCH), f"patch embed shape {tuple(w.shape)}"
    add("v.patch_embd.weight", w[:, :, 0, ...].contiguous())
    add("v.patch_embd.weight.1", w[:, :, 1, ...].contiguous())
    add("v.patch_embd.bias", get(f"{VIS}.patch_embed.proj.bias"), force_f32=True)
    n_tensors += 3

    # ---- learnable absolute position embedding table [2304, 1024] ----
    w = get(f"{VIS}.pos_embed.weight")
    assert tuple(w.shape) == (N_POS_EMBD, N_EMBD), f"pos embed shape {tuple(w.shape)}"
    if not args.raw_pos_embed:
        n_grid = int(N_POS_EMBD ** 0.5)
        assert n_grid * n_grid == N_POS_EMBD
        w = _hf_interpolate_pos_embed(w, n_grid, img_grid, img_grid)
        print(f"pos_embed: pre-interpolated {n_grid}x{n_grid} -> {img_grid}x{img_grid} "
              f"(HF bilinear, align_corners=True, no antialias)")
    add("v.position_embd.weight", w)
    n_tensors += 1

    # ---- ViT blocks ----
    for i in range(N_LAYER):
        src = f"{VIS}.blocks.{i}"
        dst = f"v.blk.{i}"
        add(f"{dst}.attn_qkv.weight", get(f"{src}.attn.qkv.weight"))
        add(f"{dst}.attn_qkv.bias", get(f"{src}.attn.qkv.bias"), force_f32=True)
        add(f"{dst}.attn_out.weight", get(f"{src}.attn.proj.weight"))
        add(f"{dst}.attn_out.bias", get(f"{src}.attn.proj.bias"), force_f32=True)
        add(f"{dst}.ffn_up.weight", get(f"{src}.mlp.linear_fc1.weight"))
        add(f"{dst}.ffn_up.bias", get(f"{src}.mlp.linear_fc1.bias"), force_f32=True)
        add(f"{dst}.ffn_down.weight", get(f"{src}.mlp.linear_fc2.weight"))
        add(f"{dst}.ffn_down.bias", get(f"{src}.mlp.linear_fc2.bias"), force_f32=True)
        add(f"{dst}.ln1.weight", get(f"{src}.norm1.weight"), force_f32=True)
        add(f"{dst}.ln1.bias", get(f"{src}.norm1.bias"), force_f32=True)
        add(f"{dst}.ln2.weight", get(f"{src}.norm2.weight"), force_f32=True)
        add(f"{dst}.ln2.bias", get(f"{src}.norm2.bias"), force_f32=True)
        n_tensors += 12

    # ---- final 2x2 merger: LayerNorm(1024) + fc1 [4096,4096] GELU + fc2 [2560,4096] ----
    add("v.post_ln.weight", get(f"{VIS}.merger.norm.weight"), force_f32=True)
    add("v.post_ln.bias", get(f"{VIS}.merger.norm.bias"), force_f32=True)
    add("mm.0.weight", get(f"{VIS}.merger.linear_fc1.weight"))
    add("mm.0.bias", get(f"{VIS}.merger.linear_fc1.bias"), force_f32=True)
    add("mm.2.weight", get(f"{VIS}.merger.linear_fc2.weight"))
    add("mm.2.bias", get(f"{VIS}.merger.linear_fc2.bias"), force_f32=True)
    n_tensors += 6

    # ---- deepstack mergers: absolute layer index {5, 11, 17} ----
    for k, layer_idx in enumerate(DEEPSTACK_IDX):
        src = f"{VIS}.deepstack_merger_list.{k}"
        dst = f"v.deepstack.{layer_idx}"
        w = get(f"{src}.norm.weight")
        assert w.shape[0] == N_EMBD * 4, f"deepstack norm dim {w.shape}"
        add(f"{dst}.norm.weight", get(f"{src}.norm.weight"), force_f32=True)
        add(f"{dst}.norm.bias", get(f"{src}.norm.bias"), force_f32=True)
        add(f"{dst}.fc1.weight", get(f"{src}.linear_fc1.weight"))
        add(f"{dst}.fc1.bias", get(f"{src}.linear_fc1.bias"), force_f32=True)
        add(f"{dst}.fc2.weight", get(f"{src}.linear_fc2.weight"))
        add(f"{dst}.fc2.bias", get(f"{src}.linear_fc2.bias"), force_f32=True)
        n_tensors += 6

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"tensors: {n_tensors}, dtype: {args.dtype}")
    print(f"wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
