# RoboRT 部署到 NVIDIA Jetson AGX Orin（交叉编译产物）

> 前置：已按 `local/agx-orin/build-cross.sh` 在容器 `robort-cross` 内完成交叉编译
> （产物在 `build-cross/bin/`，详见 `toolchain-aarch64.cmake` 与 `build-cross.sh`）。
> 本文档是 **Orin 侧部署与运行清单**（Orin 暂未接入时先备好，接入后照做）。

---

## 0. 目标板信息

| 项 | 值 |
|---|---|
| 设备 | NVIDIA Jetson AGX Orin Developer Kit（tegra234，sm_87 集成 GPU） |
| 系统 | Ubuntu 22.04.5 (jammy, arm64)，L4T 36.4.3 / JetPack 6.1 |
| 驱动 / CUDA | 540.4.0 / CUDA 12.6（板端自带，无需额外装 CUDA） |
| 访问 | `ssh baidu@192.168.1.100`（同一局域网） |

---

## 1. 产物清单（宿主编译侧）

`build-cross/bin/`（全部为 **ELF64 AArch64**，`sm_87` cubin 已编入 `libggml-cuda.so`）：

- `vla-pi05-server` — ZeroMQ/Protobuf 推理服务（运行时依赖 `libvla-pi05.so.0`、`libzmq.so.5`、`libprotobuf.so.23`、`libcudart.so.12`、`libcublas.so.12`）
- `vla-pi05-selfcheck` — 固定输入 dump 1600 floats 的自检工具（bit-exact 对比）
- `libggml*.so*`、`libmtmd.so*`、`libvla-pi05.so*` — 运行时动态库（含符号链接，需整目录拷贝）

模型/分词器（宿主机 `models/pi05_libero_finetuned_v044/`）：

- `pi05-mmproj.gguf`（832MB）、`pi05.gguf`（**6.1GB**）
- `pi05_tokenizer/`

验证工具（本仓库随附，交叉编译产物）：

- `vla-pi05-client` — 最小 REQ 客户端（本地 `local/agx-orin/`，用 `client.cpp` + `vla.proto`
  生成，交叉编译），发含一张 224x224 RGB 图的 PredictRequest 并打印 action_chunk/延迟

> ⚠️ 二进制里的 RUNPATH 指向宿主编译路径（`/mount/...`），Orin 上不存在会被加载器
> 忽略；运行时统一用 `LD_LIBRARY_PATH` 指向部署目录（见 §3）。

---

## 2. 拷贝到 Orin

```bash
# 宿主机（或容器 /mount 下等价路径）执行
ORIN=baidu@192.168.1.100
APP=/data/liuwang08/lw/lw-Infra/Robo/github/RoboRT
ssh $ORIN "mkdir -p ~/robort/bin ~/robort/models"

# 产物整目录（保留符号链接）
rsync -av $APP/build-cross/bin/ $ORIN:~/robort/bin/

# 模型 + 分词器（4.7GiB，走千兆网约 1 分钟）
rsync -av --progress $APP/models/pi05_libero_finetuned_v044/ $ORIN:~/robort/models/
```

---

## 3. Orin 侧系统依赖

运行时只需动态库（二进制链接的是 SONAME：`libzmq.so.5` / `libprotobuf.so.23`），
Ubuntu 22.04 (jammy) arm64 直接 apt 装：

```bash
sudo apt update
sudo apt install -y libzmq5 libprotobuf23
# libstdc++.so.6 / libgcc_s.so.1 / libc.so.6 系统自带；libcudart.so.12 / libcublas.so.12 由 JetPack 提供
```

验证板端 CUDA 运行时在：

```bash
ls /usr/lib/aarch64-linux-gnu/libcudart.so.12* /usr/lib/aarch64-linux-gnu/libcublas.so.12*  # 或 nvidia-l4t 路径
nvidia-smi   # 540.4.0，能列出进程即 OK
```

> 若 apt 源装不上，可直接从 JetPack 已装包
> （`dpkg -l | grep -E "cuda-cudart|libcublas"`）里拿对应 .so。

---

## 4. 运行

### 4.1 自检（先跑，确认 GPU 推理通路）

**第一步：Orin 自洽性**（同机两次必须逐字节一致，排除 CPU 兜底/非确定性）

```bash
cd ~/robort
LD_LIBRARY_PATH=$PWD/bin ./bin/vla-pi05-selfcheck \
  models/pi05-mmproj.gguf models/pi05.gguf /tmp/out_orin.txt 52
LD_LIBRARY_PATH=$PWD/bin ./bin/vla-pi05-selfcheck \
  models/pi05-mmproj.gguf models/pi05.gguf /tmp/out_orin2.txt 52
# 同机两次必须一致（同 seed 42 + 干净启动）
cmp /tmp/out_orin.txt /tmp/out_orin2.txt && echo ORIN-SELF-BITEXACT
```

**第二步：与 x86 基线容差对比**（跨架构 FP 舍入差异不可避免，用容差判据）

```bash
# 宿主机生成参考 out_ref.txt（见 local.md §4：A10, CUDA 12.6, sm_86）
# 拉回 Orin 输出后：
python3 - <<'EOF'
a = [float(x) for x in open('/tmp/out_ref.txt')]
b = [float(x) for x in open('/tmp/out_orin.txt')]
d = [abs(x - y) for x, y in zip(a, b)]
print(f"n={len(a)} max_abs_diff={max(d):.3e} mean_abs_diff={sum(d)/len(d):.3e}")
raise SystemExit(0 if max(d) < 1e-2 else 1)
EOF
echo CROSS-ARCH-OK    # max_abs_diff < 1e-2 即通过
```

> ⚠️ 自检/评测前必须 `VLA_PI05_SEED=42`（否则用 random_device，输出每次不同）。
> ⚠️ 跨架构（x86 A10 sm_86 vs Orin sm_87）CUDA kernel 指令/规约顺序不同，严格 bit-exact
> 不可能；实测 max_abs_diff≈6e-3（BF16 权重 + 50 步扩散累积），以 <1e-2 容差验收。
> 同架构回归（同 GPU）仍用严格 cmp（见 local.md §4）。

### 4.2 同步模式（基线）

```bash
cd ~/robort
VLA_PI05_SEED=42 LD_LIBRARY_PATH=$PWD/bin \
  ./bin/vla-pi05-server --bind tcp://*:5555 --timing-detail phase \
  models/pi05-mmproj.gguf models/pi05.gguf
```

### 4.3 异步 + RTC（推荐，同 x86 档位）

```bash
cd ~/robort
VLA_PI05_SEED=42 VLA_PI05_RTC=1 LD_LIBRARY_PATH=$PWD/bin \
  ./bin/vla-pi05-server --bind tcp://*:5555 --async \
  models/pi05-mmproj.gguf models/pi05.gguf
```

### 4.4 验证 CUDA 真的在跑（而非 CPU 兜底）

```bash
# 另一个终端
nvidia-smi            # 应能看到 vla-pi05-server 进程、GPU util > 0%
# server 日志应显示 ggml CUDA 后端、显存分配（pi05 权重 4.71 GiB）
```

### 4.5 Orin 本地端到端验证（已实测通过）

```bash
cd ~/robort
LD_LIBRARY_PATH=$PWD/bin ./bin/vla-pi05-client tcp://127.0.0.1:5555
# 期望：
#   request_id=42 chunk=50 action_dim=32 err=""
#   action_chunk_size=1600；latency_ms total≈888 inference≈465（首请求含 kernel 编译）
```

---

## 4A. 分布式运行：仿真在本机，推理在 Orin

架构（RoboRT server 原生支持）：Orin 跑 `vla-pi05-server`（ZeroMQ ROUTER），
本机仿真/客户端（LeRobot LIBERO 等）作为 zmq 客户端连 `tcp://<Orin>:5555`。

前置：**网络必须打通**（见下）。当前实测：宿主机(10.80.137.150) 与 Orin(192.168.1.100)
不同网段且双向不可达（宿主机出站被防火墙挡、Orin 默认网关 linkdown）；唯一通路是
中间件跳板机(172.21.145.204) 反向连回宿主机 8080 的 SSH/exec 链路，**无通用 TCP 隧道**。

打通方式（按可行性排序）：

1. **跳板机端口转发**（需跳板机 172.21.145.204 权限）：在跳板机上
   `socat TCP-LISTEN:5555,fork TCP:192.168.1.100:5555` 或
   `ssh -L 5555:192.168.1.100:5555 baidu@192.168.1.100`，并在宿主机侧放行
   出站到 `172.21.145.204:5555`。
2. **调整网络**：给 Orin 增加可达宿主机网段的路由/NAT（或把 Orin 接入能路由到
   10.80.x 的网络），使双方 IP 互通，即可直连 `tcp://192.168.1.100:5555`。
3. **反向隧道**：任一方向（宿主机侧 `ssh -R`，或 Orin 侧）建立 SSH 反向隧道，
   把 `Orin:5555` 映射到本机 `127.0.0.1:5555`。前提是某侧能出站到对方 22 端口。

打通后本机侧运行（`--vla-addr` 指向隧道/转发后的地址）：

```bash
# 例：假设用方式 1，宿主机访问 172.21.145.204:5555 转发到 Orin
CUDA_VISIBLE_DEVICES=0 VLA_PI05_SEED=42 python eval.py --vla-addr tcp://172.21.145.204:5555 ...
```

> 说明：本机侧可不用 GPU（推理全在 Orin），`vla-pi05-server` 不参与本机计算。

---

## 5. 常见问题

| 症状 | 根因/处理 |
|---|---|
| 启动报 `cannot open shared object file: libzmq.so.5` 等 | 忘设 `LD_LIBRARY_PATH=$PWD/bin`，或 Orin 上没装 §3 的 apt 包 |
| 加载模型报 CUDA 错误（如 `invalid device function` / cubin 版本） | 驱动 < 570 时加载了 CUDA 13 编的 cubin —— **必须用本仓库 CUDA 12.6 交叉产物**（`sm_87`） |
| 推理奇慢（~秒/步） | GPU 没启用，掉进 CPU 后端；确认 `nvidia-smi` 有进程、server 日志是 CUDA |
| `vla-pi05-server` 端口被占 | `pgrep -a vla-pi05-server`，先 kill 旧的 |
| RTC 无效果 | server 启动缺 `VLA_PI05_RTC=1` |

---

## 6. 备选：Orin 上原生编译

交叉产物不可用时，可在 Orin 本机直接编（JetPack 自带 CUDA 12.6 + aarch64 工具链）：

```bash
sudo apt install -y cmake pkg-config libprotobuf-dev protobuf-compiler libzmq3-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --target vla-pi05-server vla-pi05-selfcheck -j$(nproc)
```

> 集成 GPU（nvgpu）无独立显存，`GGML_CUDA_NO_VMM=ON` 避免依赖 libcuda 驱动 stub，与交叉侧一致。
