# Changelog

## [Unreleased]

### Added

- **FasterWAM 架构支持**（`src/models/fasterwam/`）
  - 全 GPU ggml graph 推理：video prefill（30 video DiT blocks）+ action denoising（20步 unrolled CUDA graph）
  - CUDA graph 加速：prefill graph + denoise graph 均在第 2 次调用后自动捕获为 CUDA graph，消除 kernel launch overhead
  - video KV cache：text/video context 不变时跳过 prefill，仅跑 7ms denoise（稳态控制频率 ~139Hz）
  - 两阶段异步接口 `prepare()` / `compute()`：服务器可流水线 prepare(N+1) 与 compute(N) 并行
  - `convert_fasterwam_to_gguf.py`：从官方 `.pt` checkpoint 转换为单文件 GGUF

- **`robo::` 公共 API**（`src/api/policy.h`）
  - 对 `vla::` 的轻量 namespace 封装，不引入额外开销
  - `robo::policy_load` / `robo::policy_free` / `robo::step` / `robo::prepare` / `robo::compute` / `robo::last_stats`
  - 测试程序 `test_fasterwam.cpp`（冒烟测试）和 `test_fasterwam_bench.cpp`（分相位 benchmark）均使用 `robo::` API

- **CMake 目标重命名**（向后兼容）
  - `robort`（库）、`robort-server`（服务进程）、`robort-selfcheck`（回归测试）
  - 旧名称 `vla-pi05` / `vla-pi05-server` / `vla-pi05-selfcheck` 通过 CMake alias / OUTPUT_NAME 保持兼容

### Performance（FasterWAM，NVIDIA A10）

| 调用场景 | 延迟（中位数）|
|---|---:|
| 稳态（text 固定，state 更新）| **7.2 ms/step** |
| text 变化（新任务）| **29.5 ms/step** |

- 相较原始 CPU 实现（~501 s/step）：稳态加速约 **70,000×**，text 变化加速约 **17,000×**

---

## 历史版本

### feat: add HY-VLA model support

- HY-VLA（腾讯 Hy-Embodied-0.5-VLA）接入 RoboRT ggml graph 推理
- 单文件 GGUF 转换脚本（双塔 VLM + 视觉塔 + flow expert + 归一化统计）
- AGX Orin 部署路径打通

### feat: async dual-thread pipeline and RTC prefix guidance for pi0.5

- async ROUTER 双线程流水线：receive 线程 prepare / compute 线程 compute，消除串行等待
- RTC（Real-Time Control）前缀导引：稳态控制频率提升至 18.7 ms/step（A10）

### feat: RoboRT initial release

- pi0.5 ggml CUDA 推理引擎
- LIBERO 评测（libero_object，10/10，seed 42）：161 ms/step（A10）
- AGX Orin 首发部署：LIBERO 全套 39/40（97.5%），104.3 ms/step
