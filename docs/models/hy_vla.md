# Hy-Embodied-0.5-VLA (HY-VLA)

腾讯 [Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)，双塔 VLM + flow expert，单文件 GGUF（无需 mmproj / tokenizer）。

## 转换

```bash
python src/models/hy_vla/convert_hy_vla_to_gguf.py \
  --ckpt <hf_ckpt_dir> --out hy_vla.gguf \
  --norm-stats <norm_stats.pkl>
```

`norm_stats.pkl` 由官方仓库 `scripts/compute_norm_robotwin.py` 生成（`--downsample-rate 3 --chunk-size 20`）；省略时使用恒等归一化。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 \
  VLA_HY_VLA_TEXT_LAYERS=32 VLA_HY_VLA_VISION_LAYERS=27 \
  ./build/bin/robort-server --bind tcp://*:5555 --timing-detail phase \
  hy_vla.gguf
```

`VLA_HY_VLA_TEXT_LAYERS` / `VLA_HY_VLA_VISION_LAYERS` 控制加载层数（发行版为 32 / 27）。
