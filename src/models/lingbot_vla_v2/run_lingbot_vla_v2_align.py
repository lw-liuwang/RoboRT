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

"""Drive the C++ (RoboRT) lingbot_vla_v2 engine with the deterministic inputs
from the HuggingFace reference export and compare every tapped intermediate.

Inputs:  the .npz produced by export_lingbot_vla_v2_reference.py.
Steps:   1. write input .npy files (clip embeddings incl. deepstack rows,
            language tokens, raw state, initial noise) for
            robort-lingbot-v2-selfcheck, which injects the vision-tower
            output directly via vla::Inputs::precomputed_img_emb;
         2. run the binary (it sets VLA_LINGBOT_V2_DUMP_DIR itself, so the
            engine dumps prefix/suffix K/V, per-layer hiddens, per-step
            suffix embeddings / velocities, final x, masks, positions ...);
         3. compare ref vs dump dir with compare_lingbot_vla_v2_ref.py.

Usage (inside lw_robort_eval container, /workspace/.venv):
  python run_lingbot_vla_v2_align.py \
    --ref     /tmp/lingbot_ref.npz \
    --mmproj  /mount/lw/github/RoboRT/local/models/lingbot-vla-v2-mmproj.gguf \
    --ckpt    /mount/lw/github/RoboRT/local/models/lingbot-vla-v2.gguf \
    --bin     /mount/lw/github/RoboRT/build-container/bin/robort-lingbot-v2-selfcheck \
    --workdir /tmp/lingbot_align
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HIDDEN = 2560
N_DEEPSTACK = 3
N_TOK = 192  # 3 views x 64 merger tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    here = Path(__file__).resolve().parent
    p.add_argument("--ref", type=Path, required=True, help="reference .npz (export_lingbot_vla_v2_reference.py)")
    p.add_argument("--mmproj", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--bin", type=Path, default=here.parents[2] / "build-container" / "bin" /
                    "robort-lingbot-v2-selfcheck")
    p.add_argument("--workdir", type=Path, default=Path("/tmp/lingbot_align"))
    p.add_argument("--tol", type=float, default=2e-2)
    args = p.parse_args()

    ref = dict(np.load(args.ref))
    indir = args.workdir / "inputs"
    outdir = args.workdir / "cpp"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. vision-tower output: [N_TOK, 2560*(1+3)] ─────────────────────
    # ref keys are per-view [n_views, 64, 2560] (batch dim stripped by the
    # export hook); flatten to [n_views*64, 2560]
    def tok(a: np.ndarray) -> np.ndarray:
        return np.asarray(a).reshape(-1, HIDDEN)

    clip = tok(ref["img.clip_out"])
    parts = [clip]
    for i in range(N_DEEPSTACK):
        parts.append(tok(ref[f"img.deepstack{i}"]))
    shapes = {tuple(a.shape) for a in parts}
    if shapes != {(N_TOK, HIDDEN)}:
        raise SystemExit(f"clip/deepstack shapes {shapes} != {(N_TOK, HIDDEN)}; "
                         "check the reference export / dump key semantics")
    img_emb = np.concatenate(parts, axis=1).astype(np.float32)  # [192, 10240]
    np.save(indir / "img_emb.npy", img_emb)

    # ── 2. tokens / state / noise ────────────────────────────────────────
    lang = np.asarray(ref["lang.tokens"]).astype(np.int32).ravel()
    np.save(indir / "lang_tokens.npy", lang)
    state = np.asarray(ref["state.raw"], dtype=np.float32).ravel()
    np.save(indir / "state.npy", state)
    noise = np.asarray(ref["noise"], dtype=np.float32).reshape(-1, 55)
    np.save(indir / "noise.npy", noise)
    print(f"[inputs] img_emb {img_emb.shape}, lang {lang.size}, state {state.size}, "
          f"noise {noise.shape} -> {indir}")

    # ── 3. run the C++ engine ───────────────────────────────────────────
    cmd = [str(args.bin), str(args.mmproj), str(args.ckpt), str(indir), str(outdir)]
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"selfcheck failed with code {r.returncode}")

    # ── 4. compare ──────────────────────────────────────────────────────
    compare = here / "compare_lingbot_vla_v2_ref.py"
    rc = subprocess.run([sys.executable, str(compare), str(args.ref), str(outdir),
                         "--tol", str(args.tol)]).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
