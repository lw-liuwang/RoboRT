#!/bin/bash
# RoboRT aarch64 交叉编译脚本（目标：NVIDIA AGX Orin / JetPack 6.x）
#
# 前置：在交叉编译容器 robort-cross 内执行（见 local/agx-orin/README.md）。
#   docker exec -it robort-cross bash
#   cd /mount/lw/lw-Infra/Robo/github/RoboRT
#   ./local/agx-orin/build-cross.sh
#
# 产物在 build-cross/bin/：vla-pi05-server、vla-pi05-selfcheck、libggml*.so、libmtmd.so、libvla-pi05.so
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT=$(pwd)
BUILD_DIR=${BUILD_DIR:-build-cross}

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/local/agx-orin/toolchain-aarch64.cmake" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_CUDA_NO_VMM=ON \
  "$@"

cmake --build "${BUILD_DIR}" --target vla-pi05-server vla-pi05-selfcheck -j"$(nproc)"

echo
echo "✅ 交叉编译完成，产物在 ${BUILD_DIR}/bin/:"
ls -la "${BUILD_DIR}/bin/"
echo
echo "⚠️  验证 CUDA 真的启用："
echo "  grep '^GGML_CUDA:' ${BUILD_DIR}/CMakeCache.txt"
echo "  file ${BUILD_DIR}/bin/vla-pi05-server      # 应为 ELF ... AArch64"
