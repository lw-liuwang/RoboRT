# FasterWAM

[FasterWAM](https://github.com/hustvl/FasterWAM)，flow-matching 动作扩散模型，无视觉塔，接收预计算 T5 文本 embedding + 机器人 state，单文件 GGUF。

## 转换

```bash
python src/models/fasterwam/convert_fasterwam_to_gguf.py \
  --ckpt <checkpoint.pt> --stats <dataset_stats.json> --out fasterwam.gguf
```

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 \
  ./build/bin/robort-server --bind tcp://*:5555 \
  fasterwam.gguf
```

## 输入格式

`precomputed_img_emb`：T5 文本 embedding，shape `[text_seq, video_text_dim]`（float32）。
`state`：归一化前的机器人本体状态，shape `[state_dim]`。
`noise`（可选）：初始 action 噪声，shape `[horizon, action_dim]`；省略时自动采样。

## 性能（NVIDIA A10）

| 场景 | 延迟 | 控制频率 |
|---|---:|---:|
| 稳态（text 固定，state 更新）| 7.2 ms/step | ~139 Hz |
| text 变化（新任务）| 29.5 ms/step | ~34 Hz |
