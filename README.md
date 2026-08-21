# RoboRT

机器人大模型推理引擎

## 环境搭建与编译

### 构建依赖
- cmake
- pkg-config
- python3
- protobuf-compiler
- libprotobuf-dev
- libzmq3-dev

### 编译（桌面端 GPU）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target vla-pi05-server vla-pi05-selfcheck -j$(nproc)
```

### 交叉编译（NVIDIA Jetson AGX Orin / aarch64）

支持交叉编译到 aarch64（目标：AGX Orin，sm_87 集成 GPU + CUDA 12.6 / JetPack 6.x），
toolchain 定义见 `local/agx-orin/toolchain-aarch64.cmake`，一键脚本见
`local/agx-orin/build-cross.sh`：

```bash
cmake -S . -B build-cross -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=local/agx-orin/toolchain-aarch64.cmake \
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON
cmake --build build-cross --target vla-pi05-server vla-pi05-selfcheck -j$(nproc)
```

产物在 `build-cross/bin/`（`vla-pi05-server`、`vla-pi05-selfcheck`、`libggml*.so`、
`libmtmd.so`、`libvla-pi05.so`，全部为 ELF AArch64）。

> ⚠️ 关键约束：必须用 CUDA 12.6 的 nvcc 交叉编译（离线编 sm_87 cubin）。
> CUDA 13 编出的 cubin 要求驱动 >= 570，而 Orin 驱动为 540.4.0，加载会失败。

### 模型转换

```bash
python src/models/pi05/convert_pi05_mmproj_to_gguf.py <hf_mmproj_dir> pi05-mmproj.gguf
python src/models/pi05/convert_pi05_to_gguf.py       <hf_ckpt_dir>    pi05.gguf
```

## 运行

```bash
CUDA_VISIBLE_DEVICES=1 VLA_PI05_SEED=42 \
  ./build/bin/vla-pi05-server --bind tcp://*:5555 --timing-detail phase \
  <pi05-mmproj.gguf> <pi05.gguf>

CUDA_VISIBLE_DEVICES=1 VLA_PI05_SEED=42 \
  ./build/bin/vla-pi05-selfcheck <pi05-mmproj.gguf> <pi05.gguf> out.txt
```

## 性能评估

### NVIDIA A10

#### pi0.5

- 评测平台：LIBERO（LeRobot 仿真）
- 模型：pi0.5（RoboRT 推理引擎）
- 数据集：libero_object
- 成功率：10/10（libero_object/task_0，seed 42）

| LeRobot 基线 | RoboRT | RoboRT Async + RTC |
|:------------------:|:------------------:|:------------------:|
| ![LeRobot 基线](docs/figs/lerobot.gif) | ![RoboRT](docs/figs/robort.gif) | ![RoboRT Async + RTC](docs/figs/async_rtc.gif) |
| **377 ms/step** | **161 ms/step** | **18.7 ms/step** |

### NVIDIA Jetson AGX Orin

#### pi0.5

- 评测平台：LIBERO（LeRobot 仿真，经网络连接 AGX Orin 推理）
- 模型：pi0.5（RoboRT 推理引擎，`--async` + `VLA_PI05_RTC=1`）
- 数据集：libero_10 / libero_goal / libero_object / libero_spatial（40 任务）
- 成功率：**39/40（97.5%）**
- 平均 infer：**104.3 ms/step**

| suite | 成功率 | 平均 infer |
|---|---:|---:|
| libero_10 | 9/10 | 101.1 ms/step |
| libero_goal | 10/10 | 105.9 ms/step |
| libero_object | 10/10 | 102.8 ms/step |
| libero_spatial | 10/10 | 107.4 ms/step |
| **总计** | **39/40（97.5%）** | **104.3 ms/step** |

| libero_goal | libero_object | libero_spatial |
|:------------------:|:------------------:|:------------------:|
| ![libero_goal](docs/figs/orin_libero_goal.gif) | ![libero_object](docs/figs/orin_libero_object.gif) | ![libero_spatial](docs/figs/orin_libero_spatial.gif) |

## 技术报告

RoboRT: A Real-Time C++ Inference Engine for pi0.5 Vision-Language-Action Policies（[PDF](docs/paper/robort-tech-report.pdf)，[LaTeX 源码](docs/paper/robort-tech-report.tex)）

## License

Apache-2.0（见 `LICENSE`）。`third_party/` 下第三方组件版权归各自作者，许可与版权声明见 `third_party/NOTICE`。
