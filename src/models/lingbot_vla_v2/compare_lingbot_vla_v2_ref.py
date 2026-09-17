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

"""Compare two .npz dumps (HF reference vs C++ engine) for lingbot_vla_v2
numerical alignment.

Keys are matched exactly (both sides use the same naming; the C++ dump uses
'.' separators, the reference uses '/' which this script normalises).  For
each common key it reports max-abs / mean-abs / rel error and a cosine
similarity; mismatched key sets are listed.

Usage:
  python compare_lingbot_vla_v2_ref.py ref.npz cpp.npz [--tol 2e-2] [--prefix s0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def key_norm(k: str) -> str:
    return k.replace("/", ".")


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"shape_mismatch": (a.shape, b.shape)}
    d = a - b
    denom = max(np.abs(a).max(), 1e-12)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "rel_max": float(np.abs(d).max() / denom),
        "cos": cos,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("cpp", type=Path)
    p.add_argument("--tol", type=float, default=2e-2,
                   help="rel_max threshold for PASS (bf16-level agreement)")
    p.add_argument("--prefix", default="",
                   help="only compare keys containing this substring")
    args = p.parse_args()

    ref = {key_norm(k): v for k, v in np.load(args.ref).items()}
    if args.cpp.is_dir():
        # Alignment-driver output: one .npy per key (file name == key).
        cpp = {p.stem: np.load(p) for p in sorted(args.cpp.glob("*.npy"))}
    else:
        cpp = {key_norm(k): v for k, v in np.load(args.cpp).items()}

    # ── lingbot-specific normalisation ──────────────────────────────────
    # 1. the HF reference pads language to tokenizer_max_length; the C++
    #    engine omits those masked pad tokens (they are excluded from every
    #    attention row, so the math is identical).  Strip pad columns from
    #    any ref array axis of size P (prefix) or P+S (prefix+suffix KV).
    if "prefix.pad" in ref:
        keep = np.asarray(ref["prefix.pad"]).ravel() == 1
        P = keep.size
        S = int(np.asarray(ref["s0.pos"]).shape[-1]) if "s0.pos" in ref else 51
        keep_full = np.concatenate([keep, np.ones(S, dtype=bool)])

        def strip(a: np.ndarray) -> np.ndarray:
            out = a
            for ax, n in enumerate(out.shape):
                if n == P:
                    out = np.compress(keep, out, axis=ax)
                elif n == P + S:
                    out = np.compress(keep_full, out, axis=ax)
            return out

        ref = {k: (strip(v) if isinstance(v, np.ndarray) and v.ndim else v)
               for k, v in ref.items()}
        print(f"[strip] removed {int((~keep).sum())} lang pad tokens "
              f"(P {P} -> {int(keep.sum())})")
    # 2. final action key naming
    if "final.action" in ref and "final.actions_raw" not in ref:
        ref["final.actions_raw"] = ref.pop("final.action")
    # 3. prefix.deepstack{l}: the C++ engine dumps a whole-prefix [P, hidden]
    #    scatter map (features only at image-token rows); the reference stores
    #    the per-view features [3, 64, hidden].  Extract the image rows.
    #    Prefix layout: per view [vs, 64 img tokens, ve] = 66 rows.
    for k in [k for k in list(cpp) if k.startswith("prefix.deepstack")]:
        a = np.asarray(cpp[k])
        if a.ndim == 2 and a.shape[1] == 2560:
            rows = []
            for v in range(3):
                base = v * 66
                rows.extend(range(base + 1, base + 65))
            cpp[k] = a[rows].reshape(3, 64, 2560)

    keys = sorted(set(ref) & set(cpp))
    if args.prefix:
        keys = [k for k in keys if args.prefix in k]
    only_ref = sorted(set(ref) - set(cpp))
    only_cpp = sorted(set(cpp) - set(ref))

    n_pass = n_fail = 0
    worst: list[tuple[float, str]] = []
    print(f"{'key':48s} {'shape':20s} {'max_abs':>10s} {'rel_max':>10s} {'cos':>8s}")
    for k in keys:
        s = stats(ref[k], cpp[k])
        if "shape_mismatch" in s:
            print(f"{k:48s} SHAPE {s['shape_mismatch'][0]} vs {s['shape_mismatch'][1]}")
            n_fail += 1
            continue
        ok = s["rel_max"] <= args.tol or s["cos"] >= 0.9999
        n_pass += ok
        n_fail += not ok
        worst.append((s["rel_max"], k))
        flag = "ok " if ok else "FAIL"
        print(f"{k:48s} {str(ref[k].shape):20s} {s['max_abs']:10.3e} "
              f"{s['rel_max']:10.3e} {s['cos']:8.5f} {flag}")

    print(f"\n{ n_pass }/{n_pass + n_fail} keys within tol (rel_max<={args.tol:g})")
    if worst:
        print("worst 10 by rel_max:")
        for r, k in sorted(worst, reverse=True)[:10]:
            print(f"  {r:10.3e}  {k}")
    if only_ref:
        print(f"only in ref ({len(only_ref)}):", only_ref[:10])
    if only_cpp:
        print(f"only in cpp ({len(only_cpp)}):", only_cpp[:10])


if __name__ == "__main__":
    main()
