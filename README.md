# RoboRT

机器人大模型推理引擎（Robot Brain Inference Engine）

支持的架构：
- **pi0.5**（Physical Intelligence）
- **HY-VLA**（腾讯 Hy-Embodied-0.5-VLA）
- **FasterWAM**（流匹配动作扩散，纯动作推理，无视觉塔）

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
cmake --build build --target robort-server robort-selfcheck -j$(nproc)
```

> `vla-pi05-server` / `vla-pi05-selfcheck` 作为别名仍然有效（向后兼容）。

### 模型转换

**pi0.5**

```bash
python src/models/pi05/convert_pi05_mmproj_to_gguf.py <hf_mmproj_dir> pi05-mmproj.gguf
python src/models/pi05/convert_pi05_to_gguf.py       <hf_ckpt_dir>    pi05.gguf
```

**HY-VLA**（腾讯 Hy-Embodied-0.5-VLA）单文件 GGUF（双塔 VLM + 视觉塔 + flow expert + 归一化统计）：

```bash
python src/models/hy_vla/convert_hy_vla_to_gguf.py \
  --ckpt <hf_ckpt_dir> --out hy_vla.gguf \
  --norm-stats <norm_stats.pkl>   # 可选；缺失时用恒等归一化
```

- `--norm-stats`：由官方仓库 `scripts/compute_norm_robotwin.py` 产出（`--downsample-rate 3 --chunk-size 20`）

**FasterWAM**（hustvl/FasterWAM）单文件 GGUF：

```bash
python src/models/fasterwam/convert_fasterwam_to_gguf.py \
  --ckpt <checkpoint.pt> --stats <dataset_stats.json> --out fasterwam.gguf
```

## 运行

**pi0.5**

```bash
CUDA_VISIBLE_DEVICES=0 VLA_PI05_SEED=42 \
  ./build/bin/robort-server --bind tcp://*:5555 --timing-detail phase \
  <pi05-mmproj.gguf> <pi05.gguf>
```

**HY-VLA**（`robort-server` 按 GGUF 内 `hy_vla.architecture` 自动分派）：

```bash
CUDA_VISIBLE_DEVICES=0 VLA_HY_VLA_TEXT_LAYERS=32 VLA_HY_VLA_VISION_LAYERS=27 \
  ./build/bin/robort-server --bind tcp://*:5555 --timing-detail phase \
  <hy_vla.gguf>
```

> `VLA_HY_VLA_TEXT_LAYERS` / `VLA_HY_VLA_VISION_LAYERS` 控制加载层数（发行版为 32 / 27）。

**FasterWAM**（`robort-server` 按 `fasterwam.architecture` 自动分派）：

```bash
CUDA_VISIBLE_DEVICES=0 \
  ./build/bin/robort-server --bind tcp://*:5555 \
  <fasterwam.gguf>
```

## 性能评估

### NVIDIA Jetson AGX Orin

#### pi0.5

- 评测平台：LIBERO（LeRobot 仿真，经网络连接 AGX Orin 推理）
- 模型：pi0.5（RoboRT，`--async` + `VLA_PI05_RTC=1`）
- 数据集：libero_10 / libero_goal / libero_object / libero_spatial（40 任务）
- 成功率：**39/40（97.5%）**，平均推理：**104.3 ms/step**

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

#### HY-VLA

- 支持状态：已打通 Orin 部署路径（`VLA_HY_VLA_TEXT_LAYERS=32 VLA_HY_VLA_VISION_LAYERS=27`）
- LIBERO 评测结果：待补充

### NVIDIA A10

#### pi0.5

- 数据集：libero_object，成功率：10/10（task_0，seed 42）

| LeRobot 基线 | RoboRT | RoboRT Async + RTC |
|:------------------:|:------------------:|:------------------:|
| ![LeRobot 基线](docs/figs/lerobot.gif) | ![RoboRT](docs/figs/robort.gif) | ![RoboRT Async + RTC](docs/figs/async_rtc.gif) |
| **377 ms/step** | **161 ms/step** | **18.7 ms/step** |

#### FasterWAM（NVIDIA A10，RoboTwin）

全 GPU ggml graph 推理，CUDA graph 加速：

| 调用场景 | 延迟（中位数）| 控制频率 |
|---|---:|---:|
| 稳态（text 不变，state 更新）| **7.2 ms/step** | ~139 Hz |
| text 变化（新任务）| **29.5 ms/step** | ~34 Hz |

- video prefill（text 变时）：22 ms（30 video DiT blocks，A10 内存带宽极限）
- action denoising（每步）：7 ms（20步 unrolled CUDA graph，30 action DiT blocks）
- 相较原始 CPU 实现（~501s/step）加速约 **17,000–70,000×**

## C++ API

`policy.h` 提供 `robo::` 命名空间（对 `vla::` 的轻量封装）：

```cpp
#include "policy.h"

robo::Policy * m = robo::policy_load("", "fasterwam.gguf", "");
const robo::PolicyConfig & cfg = robo::policy_config(m);

robo::PolicyInput in{};
in.state = state_data;
in.noise = noise_data;

// 单步推理
std::vector<float> actions = robo::step(m, in);

// 两阶段异步推理（服务器流水线）
robo::PreparedStep ps = robo::prepare(m, in);   // phase 1：prefill
std::vector<float> actions = robo::compute(m, ps); // phase 2：denoise

robo::policy_free(m);
```

## 技术报告

RoboRT: A Real-Time C++ Inference Engine for pi0.5 Vision-Language-Action Policies（[PDF](docs/paper/robort-tech-report.pdf)，[LaTeX 源码](docs/paper/robort-tech-report.tex)）

## License

Apache-2.0（见 `LICENSE`）。`third_party/` 下第三方组件版权归各自作者，许可与版权声明见 `third_party/NOTICE`。
