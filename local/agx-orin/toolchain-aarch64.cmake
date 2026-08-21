# RoboRT aarch64 (NVIDIA AGX Orin / JetPack 6.x) 交叉编译 toolchain
#
# 目标：AGX Orin —— aarch64 + sm_87 集成 GPU + CUDA 12.6 (JetPack 6.1, 驱动 540.4.0)
#
# ⚠️ 关键约束：
#   - 必须用 CUDA 12.6 的 nvcc（x86 宿主上的工具链即可，离线编 sm_87 cubin）。
#     CUDA 13 编出的 cubin 要求驱动 >= 570，而 Orin 驱动是 540.4.0，加载会失败。
#   - 宿主编译在容器 robort-cross 内进行（Ubuntu 22.04 + CUDA 12.6 nvcc + aarch64 gcc）。
#
# 用法（见 local/agx-orin/build-cross.sh）：
#   cmake -S . -B build-cross \
#     -DCMAKE_TOOLCHAIN_FILE=local/agx-orin/toolchain-aarch64.cmake \
#     -DGGML_CUDA=ON

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# CUDA 12.6 nvcc（x86 宿主工具，生成 sm_87 设备代码）
set(CMAKE_CUDA_COMPILER      /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CUDA_ARCHITECTURES 87)

# 非 CUDA 的 aarch64 系统库（libzmq / libprotobuf 及闭包），由 Ubuntu 22.04 (jammy) arm64
# deb 解包生成（glibc 2.35 / libstdc++ 11，与 Orin 上的 Ubuntu 22.04.5 一致）。
set(SYSROOT /mount/lw/lw-Infra/Robo/aarch64-sysroot)
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CUDA toolkit 根：容器内 CUDA 12.6。nvcc 在 bin/（x86），aarch64 库在
# targets/aarch64-linux（符号链接 -> sbsa-linux，由 arm64 CUDA deb 解包而来）。
# CUDAToolkit 模块在交叉模式下按 CMAKE_SYSTEM_PROCESSOR 自动选择 targets/aarch64-linux。
set(CUDAToolkit_ROOT /usr/local/cuda-12.6 CACHE PATH "CUDA 12.6 toolkit root (x86 nvcc + aarch64 libs)")

# pkg-config 只查 sysroot 内的 .pc（如 libzmq.pc），且给返回路径加 sysroot 前缀
set(ENV{PKG_CONFIG_LIBDIR}      ${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig)
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${SYSROOT})

# 链接最终可执行文件时，ld 需解析 libggml-cuda.so 的传递依赖 libcudart.so.12 /
# libcublas.so.12。这些 aarch64 库不在 sysroot 里，显式加 -L（无 CMAKE_SYSROOT，
# 绝对路径原样传给 ld）。
set(CUDA_AARCH64_LIB ${CUDAToolkit_ROOT}/targets/sbsa-linux/lib)
# 注意：CMAKE_*_LINKER_FLAGS 首次 configure 会写进缓存，必须用 FORCE 覆盖，
# 否则第二次起 configure 时新加的 flags 不生效。
# -L 只对命令行里的库生效；解析 libggml-cuda.so 的 DT_NEEDED（libcudart.so.12 /
# libcublas.so.12）必须用 -rpath-link（GNU ld 不拿 -L 目录去解析传递依赖）。
set(CMAKE_EXE_LINKER_FLAGS    "-L${CUDA_AARCH64_LIB} -Wl,-rpath-link,${CUDA_AARCH64_LIB}"    CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-L${CUDA_AARCH64_LIB} -Wl,-rpath-link,${CUDA_AARCH64_LIB}" CACHE STRING "" FORCE)
