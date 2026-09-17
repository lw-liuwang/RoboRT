# LingBot-VLA-v2-6B

东南大学 [LingBot-VLA-V2](https://modelscope.cn/collections/Robbyant/LingBot-VLA-V2)（RoboTwin 具身），Qwen3-VL-4B 双塔 + Qwen2 MoE flow-matching 动作专家，联合注意力（suffix 直接拼 prefix KV），RoboRT 以双文件 GGUF（mmproj + 主模型）格式加载。

## 前置

- checkpoint：`lingbot-vla-v2-6b`（6 个 safetensors 分片）
- norm stats：`local/lingbot-vla-v2/assets/norm_stats/robotwin.json`（在 HF 源码仓库内，ModelScope checkpoint 不含此文件）
- HF 源码仓库（含 configs 与 norm stats）：`local/lingbot-vla-v2`（对齐脚本需要，可 `git clone` 官方 repo）
- Qwen3-VL-4B-Instruct 基座的 config / processor / tokenizer 小文件（无权重）：`local/models/Qwen3-VL-4B-Instruct`
- 转换与对齐需在带 `torch` + `transformers` + `gguf` 包的环境运行（容器 `lw_robort_eval` 的 `/workspace/.venv`）

## 转换

```bash
# 视觉塔（Qwen3-VL + qwen3vl_merger，输出含 deepstack 特征 10240 维/token）
python src/models/lingbot_vla_v2/convert_lingbot_mmproj_to_gguf.py \
  --ckpt local/models/lingbot-vla-v2-6b --out local/models/lingbot-vla-v2-mmproj.gguf

# 主模型（VLM 36 层 + MoE expert 36 层 + flow-matching 头 + norm stats）
python src/models/lingbot_vla_v2/convert_lingbot_vla_v2_to_gguf.py \
  --ckpt local/models/lingbot-vla-v2-6b \
  --stats local/lingbot-vla-v2/assets/norm_stats/robotwin.json \
  --out local/models/lingbot-vla-v2.gguf
```

`--dtype bf16` 可将 matmul 权重压半（norm/bias/router 恒 F32）。默认 f32 精度最高。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 \
  ./build-container/bin/robort-server --bind tcp://*:5555 --timing-detail phase \
  lingbot-vla-v2-mmproj.gguf lingbot-vla-v2.gguf
```

环境变量：

| 变量 | 说明 |
|---|---|
| `VLA_LINGBOT_V2_FLASH=1` | attention 走 `ggml_flash_attn_ext`（CUDA FA）。默认关 |
| `VLA_LINGBOT_V2_FA_F16=0` | 仅 FA 时有效：联合 K/V 缓存退回 F32。默认 F16 |
| `VLA_LINGBOT_V2_FA_COPY=1` | 仅 FA 时有效：物化 Q/K/V 的 `permute`，默认靠 kernel stride 直接读 |
| `VLA_LINGBOT_V2_NUM_STEPS=N` | 去噪步数（默认 10，有损：改的是推理轨迹） |
| `VLA_LINGBOT_V2_SEED=N` | 确定性初始噪声 |
| `VLA_LINGBOT_V2_F32_WEIGHTS` | 全部 matmul 权重按 F32 加载（覆盖 bf16 转换） |
| `VLA_LINGBOT_V2_NO_CACHE=1` | 每次 predict 重建计算图 |
| `VLA_LINGBOT_V2_DUMP_DIR=path` | 数值对齐 dump（见下）。dump 时 FA 的 K/V 缓存自动退回 F32 拼接路径（dump 要 tap 拼接后的联合 K/V） |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 关闭 CUDA graphs（基线的 A/B 对照） |

## 无损加速

三项相互叠加，全部保持 action chunk 逐位不变（同二进制 + 固定 noise 逐位比对验证）：

1. **CUDA graphs**（构建期开关，`build-container` 已置 ON）。vendored ggml 原本因为
   `GGML_OP_MUL_MAT_ID` 可能 sync stream 而拒绝捕获整个 lingbot 图，这里补了兼容性判定：
   非量化 `mul_mat_id` 走 mmf 融合核时不会 sync（判据与 `ggml_cuda_mul_mat_id` 的 dispatch
   一致，见 `ggml-cuda.cu` 的 `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`）。整张 28k 节点图捕获成一次发射。
   注意前 2 次请求用于图捕获（~380 ms），测延迟前需先 warmup。
2. **FA 去拷贝**（默认生效）。CUDA FA kernel 直接按 `nb[1]/nb[2]` 读 Q/K/V，FA 分支里
   `ggml_cont(ggml_permute(...))` 的三个拷贝纯属记账搬运。
3. **F16 联合 K/V 缓存**（默认生效，`VLA_LINGBOT_V2_FA_F16=0` 关）。`launch_fattn` 的
   `to_fp16` 与 `ggml_cast` 走同一个 `ggml_cuda_cast<half>(float)`，所以缓存直接以 F16 保存
   逐位等价；这样 230-token 前缀的 F32→F16 转换从"每层每步"降为"每层一次"，每步也不再
   物化 prefix+suffix 的全长拼接（改为 cpy 写入缓存尾部）。

A10 / bf16 权重 / 20 次请求中位（vision 不计入 inference）：

| 配置 | inference | wall |
|---|---|---|
| 基线（无 graphs、无 FA） | 259.5 ms | 305 ms |
| + CUDA graphs | 240.9 ms | 284 ms |
| + FA | 231.6 ms | 275 ms |
| + FA 去拷贝 | 212.7 ms | 254 ms |
| + F16 K/V 缓存 | **210.9 ms** | 254 ms |

复现方式：固定输入（客户端显式传 `noise`，或固定 `VLA_LINGBOT_V2_SEED` 且同一个请求），
重复 6 次比较 `action_chunk` 的 sha256（同一配置内必须完全一致），再在同一二进制上切换上面的
env 对比两条腿的 sha256（必须相同）。注意 sha256 的绝对值取决于客户端构造的输入，只能作
同客户端两腿之间的判据。

## 延迟构成（2026-09-16 复测）

`--timing-detail phase` 只对 `vision` 与 `inference` 有意义：整张图（prefix + 10 步去噪）是
**一次 `ggml_backend_graph_compute`**，引擎填不了 `latency_ms_prefill/denoise`（这两个字段
在 `api/model.h` 里有，但没有模型实现，故恒为 0）。拆分靠 `VLA_LINGBOT_V2_NUM_STEPS`：
`inf(N) - inf(1) = (N-1) × 每步`，`prefix ≈ inf(1) - 每步`。

A10 / bf16 权重 / `VLA_LINGBOT_V2_FLASH=1` / n_lang=16（prefix 230 token）：

| 阶段 | ms | 占比 |
|---|---|---|
| vision（独立 766 节点图，3 张图各跑一次） | 32.9 | 13% |
| prefix（VLM 36 层，230 token） | 47.8 | 19% |
| denoise（10 步 × 专家塔 36 层，51 token，k_len 281） | 163.8 | 65% |
| 主机/zmq 等 | ~7 | 3% |
| **每请求 total** | **~252** | |

每步 16.4 ms = 36 层 × 0.455 ms。**跨进程噪声 ±3~4 ms**，小差异必须在同一进程内 A/B 才算数
（例：n_lang 16→72 在同一进程内是 +9.1 ms）。

推算的权重流量（每请求）：专家塔 ~30 GB/10 步（routed MoE 27.2 GB），prefix ~7.3 GB，
vision ~3.6 GB → 合计 ~41 GB / 0.25 s ≈ 164 GB/s，约 A10 峰值 600 GB/s 的 27%。也就是说
denoise 既没跑到带宽上限、更没跑到算力上限，卡在"极小 kernel 的地址运算/装箱"上。

## 进一步加速（全部有损，2026-09-16 实测）

无损空间已经花完（−18.7%，全链路 bit-exact）。再往下必须动推理语义，实测三条：

| 手段 | 收益 | 代价 |
|---|---|---|
| `VLA_LINGBOT_V2_NUM_STEPS=5` | inf 214→132 ms、total 256→171 ms（−33%） | 固定 noise 下 action_chunk max\|d\|=0.326、mean 0.054（= 参考 std 的 0.42σ） |
| 专家塔权重 W8 量化 | denoise（65%）主体流量减半，预估 −40~60 ms | 需实现 + 重跑评测验证精度 |
| 请求级 batch（多路并发共享一次 `mul_mat_id`） | 吞吐 2~4×（每步仅 51 token，权重复用极差） | 改动大（变长 prefix + 块对角 mask）；可能保持 bit-exact |

否决过：`MTMD_VISION_BF16=1`（`clip.cpp` 的 env 逃生口）对 lingbot 的 ViT **无效**——vision
33.2→33.7 ms（噪声内），且 `action_chunk` 与 F32 逐位相同。ViT 权重在 mmproj 里是 F32 且不是
bf16 上采样（316/316 张量低位非零），压 bf16 属于有损且无收益。

## 输入格式

- 图片：3 路 256×256（cam_high / 左右腕）；内部按 Qwen 归一化（`u8/127.5-1`）送视觉塔，每图 64 token。
- 语言：Qwen tokenizer id 序列（`lang_tokens`）。
- `state`：原始机器人 14 维（RoboTwin [armL6, gripL, armR6, gripR]），引擎内部做 canonical 55 维 scatter + `bounds_99_woclip` 归一化。
- `noise`（可选）：`[50, 55]`（token × dim）初始噪声；省略时模型 RNG 采样。
- 输出：`[50, 14]` 去归一化动作块。
- 也支持 `precomputed_img_emb`：`[192, 10240]`（每 token 前 2560 为 merger 嵌入，后 3×2560 为 deepstack ViT-5/11/17 特征）。

## 数值对齐

三件套（均需参考侧 HF 推理，绕开 flash_attn：joint attention=eager、ViT=sdpa）：

```bash
# 1. HF 参考中间结果（容器内 /workspace/.venv）
python src/models/lingbot_vla_v2/export_lingbot_vla_v2_reference.py \
  --ckpt  local/models/lingbot_vla-v2-6b \
  --qwen3vl local/models/Qwen3-VL-4B-Instruct \
  --out /tmp/lingbot_ref.npz

# 2. 用参考输入驱动 C++ 引擎并 dump 全部中间量（KV、逐层 hidden、逐步 suffix/v、masks、pos…）
python src/models/lingbot_vla_v2/run_lingbot_vla_v2_align.py \
  --ref /tmp/lingbot_ref.npz \
  --mmproj local/models/lingbot-vla-v2-mmproj.gguf \
  --ckpt local/models/lingbot-vla-v2.gguf \
  --bin build-container/bin/robort-lingbot-v2-selfcheck \
  --workdir /tmp/lingbot_align

# 3. 逐步骤比对（自动在步骤 2 末尾执行；也可单独运行）
python src/models/lingbot_vla_v2/compare_lingbot_vla_v2_ref.py \
  /tmp/lingbot_ref.npz /tmp/lingbot_align/cpp --tol 2e-2
```

比对工具支持 ref `.npz` vs C++ dump 目录（`.npy` per key）或另一 `.npz`；判据 `rel_max<=tol` 或 `cos>=0.9999`（bf16 级一致）。

### 对齐结果（f32 权重、CPU、随机确定性输入，2026-09-15）

245 个公共键中 240 个 PASS：clip/deepstack/查询 bank/位置/mask 精确一致（0.0），
prefix 全部 KV/逐层 hidden、`prefix.out_vlm`、suffix 嵌入与 expert 塔逐层输入
`rel_max<=~5e-3`（fp32 op 顺序差异），**`final.actions_raw` rel_max 2.3e-2 / cos 0.99997**
（端到端动作输出对齐）。仅 5 个中间速度 `s{2,5,7,8,9}.v` 超阈（rel_max 4–13%,
cos>=0.9994）：fp32 运算顺序差异使 MoE top-4 路由在个别 borderline token 上翻转（离散效应，
误差集中在 ~8/50 行），经 10 步 Euler 累积放大；终点动作仍在对齐判据内，无需修复。

参考侧已知语义差异（compare 脚本自动归一化）：
1. **lang padding**：HF 参考把语言 pad 到 `tokenizer_max_length=72`（56 个 pad token，
   所有注意力行中均被 mask）；C++ 引擎直接省略 pad（数学等价），compare 按 `prefix.pad`
   剔除参考侧 pad 轴。
2. **prefix.deepstack\***：C++ dump 的是全 prefix [P,2560] 散射图（仅 img 行非零），
   参考存逐视图 [3,64,2560]；compare 提取 img 行（每视图 66 token 中的 1..64）。

导出脚本踩过的坑（已修复，复现时注意）：`put()` 若不 `clone()`，fp32 CPU 张量的
`.float().cpu().numpy()` 会**共享原张量内存**——denoise 循环 `x_t += dt·v` 原地改写
noise 张量后，`noise`/`s*.x_in`/`final.x` 等 dump 被静默改写成循环终值。另外 `noise`
快照必须放在 `sample_actions` 调用**之前**（x_t 与 noise 同一对象）。

## 架构要点（适配备忘）

- 联合注意力：prefix pass（VLM 塔，causal）填 KV cache；每步 denoise 时 expert 塔 51 token（1 state + 50 action）的 K/V 拼接到 prefix KV 后做单次 attention。两塔同几何（32Q/8KV/128hd），统一 Qwen3-VL interleaved M-RoPE（sections [24,20,20]）。
- expert 塔每层：AdaRMSNorm(x, t_emb) → Qwen2 attn（q/k/v 带 bias、无 QK-norm）→ MoE（sigmoid router、top-k=4、norm_topk_prob、routed_scaling=4.0）+ **ungated shared expert**（robotwin.yaml `use_shared_expert_gate=false`，checkpoint 无 gate 权重）。
- Deepstack：clip 输出 10240/token，ViT-5/11/17 特征注入 VLM 层 0/1/2 输出的 image token 位置。
- Prefix：3×(vs + 64 img + ve) + lang + 8 current + 8 future 任务查询（查询 bank 在转换时预融合进 GGUF）。
- 采样：10 步 Euler（t=1→0.1），`x_t += dt·v`，`denorm=(x+1)·(q99-q01+1e-6)·0.5+q01`。
