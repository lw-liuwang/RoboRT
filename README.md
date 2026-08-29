<div align="center">

<img src="docs/figs/robort_icon.png" width="120"/>

# RoboRT

**A Robot Model Inference Engine**

</div>

---

## 支持的模型

| 模型 | 硬件 | 推理延迟 | 参考基线 | 加速倍数 |
|---|---|---:|---:|---:|
| [$\pi_{0.5}$](docs/models/pi05.md) | AGX Orin | **104.3 ms/step** | — | — |
| [$\pi_{0.5}$](docs/models/pi05.md) | A10 | **18.7 ms/step** (Async+RTC) | 377 ms (LeRobot) | **~20×** |
| [Hy-Embodied-0.5-VLA](docs/models/hy_vla.md) | A10 | — | — | — |
| [FasterWAM](docs/models/fasterwam.md) | A10 | **7.2 ms/step** (稳态) | ~501,000 ms (CPU) | **~70,000×** |

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target robort-server -j$(nproc)
```

依赖：`cmake` `pkg-config` `protobuf-compiler` `libprotobuf-dev` `libzmq3-dev`

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 \
  ./build/bin/robort-server --bind tcp://*:5555 \
  [mmproj.gguf] policy.gguf
```

模型转换与模型特定参数见 [docs/models/](docs/models/)。

## C++ API

```cpp
#include "policy.h"

robo::Policy * m = robo::policy_load("", "policy.gguf");

robo::PolicyInput in{};
in.state = state_data;
std::vector<float> actions = robo::step(m, in);

robo::policy_free(m);
```

## 性能

### NVIDIA Jetson AGX Orin — $\pi_{0.5}$ on LIBERO

| suite | 成功率 | infer |
|---|---:|---:|
| libero_10 | 9/10 | 101.1 ms/step |
| libero_goal | 10/10 | 105.9 ms/step |
| libero_object | 10/10 | 102.8 ms/step |
| libero_spatial | 10/10 | 107.4 ms/step |
| **总计** | **39/40 (97.5%)** | **104.3 ms/step** |

| libero_goal | libero_object | libero_spatial |
|:---:|:---:|:---:|
| ![libero_goal](docs/figs/orin_libero_goal.gif) | ![libero_object](docs/figs/orin_libero_object.gif) | ![libero_spatial](docs/figs/orin_libero_spatial.gif) |

### NVIDIA A10 — $\pi_{0.5}$ on LIBERO

| LeRobot 基线 | RoboRT | RoboRT Async + RTC |
|:---:|:---:|:---:|
| ![LeRobot](docs/figs/lerobot.gif) | ![RoboRT](docs/figs/robort.gif) | ![Async+RTC](docs/figs/async_rtc.gif) |
| 377 ms/step | 161 ms/step | **18.7 ms/step** |

### NVIDIA A10 — FasterWAM on RoboTwin

| 场景 | 延迟 |
|---|---:|
| 稳态（text 固定） | **7.2 ms/step** |
| text 变化 | **29.5 ms/step** |

## License

Apache-2.0。`third_party/` 下各组件版权归各自作者，见 `third_party/NOTICE`。
