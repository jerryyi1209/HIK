# CarpetSegNet v2

Multi-modal carpet segmentation network fusing RGB, dense depth, and geometric features. Designed for Intel RealSense D435 on mobile robots.

![Architecture](https://img.shields.io/badge/PyTorch-2.4.1-red) ![Backbone](https://img.shields.io/badge/Backbone-ResNet50-blue) ![Input](https://img.shields.io/badge/Input-9ch-green) ![mIoU](https://img.shields.io/badge/Best_mIoU-0.8272-brightgreen)

## Highlights

- **9-channel input**: RGB(3) + Depth(1) + Surface Normals(3) + Depth Edge(1) + Curvature(1)
- **28M params**, ResNet50 backbone, U-Net decoder with RefinementHead
- **Best mIoU: 0.8272** on NYU Depth V2 carpet/floor benchmark
- **Jetson Orin Nano** deployment ready (ONNX → TensorRT pipeline)

## Quick Start

```bash
# Python 3.8+, PyTorch 2.4+, CUDA GPU
pip install -r requirements.txt

# Verify setup (synthetic tensors, no real data)
python scripts/test_model.py

# GPU diagnostic
python scripts/gpu_diag.py --quick
```

### Inference

```bash
# Single image
python scripts/infer_deploy.py \
    --checkpoint experiments/carpet_seg0807/checkpoints/best_mIoU-0-8272_20260807_205623_deploy.pth \
    --rgb rgb.png --depth depth.png \
    --output mask.png
```

See [deploy/README.md](deploy/README.md) for Jetson deployment, ONNX export, and real-time D435 inference.

## Training

### Data Preparation

```bash
# 1. Convert NYU .mat → frames
python scripts/convert_nyu.py

# 2. Filter carpet/floor frames
python scripts/filter_nyu.py

# 3. Offline augmentation (175 pos → 875 variants)
python scripts/augment_positives.py --num-variants 4
```

Dataset: 1,575 train / 187 val / 189 test from NYU Depth V2.

### Train

```bash
python scripts/train_v2.py --exp carpet_seg_v2
python scripts/train_v2.py --exp carpet_seg_v2 --resume experiments/carpet_seg_v2/checkpoints/last.pth
```

## Architecture

```
RGB(3) + Depth(1) + GeoFeature(5) = 9ch
         │
    SimpleEncoder (ResNet50, expanded first conv)
         │
    [f1, f2, f3, f4] — 4 scales
         │
    SimpleDecoder (U-Net + lateral skip connections)
         │
    RefinementHead (stride 4→2→1, 2× resolution)
         │
    logits (2, H, W)
```

### Loss Function

```
Loss = CE(edge-weighted, label_smoothing=0.1) + Dice + 0.5 × Focal(γ=2)
```

- **Edge-aware weighting**: Distance-transform boundary emphasis
- **Label smoothing**: Prevents overconfidence on small dataset
- **Balanced sampling**: 50/50 pos/neg per batch

## Results

| Experiment | Best mIoU | Key Features |
|---|---|---|
| `carpet_seg0807` | **0.8272** | Label smoothing + weight_decay=5e-4 + 9ch geo |
| `carpet_seg0805` | 0.8014 | 9ch geo baseline |
| `carpet_seg_d435_v2` | 0.7946 | First 9ch run |

## Deployment

[Jetson Orin Nano](deploy/README.md) with ONNX → TensorRT pipeline for real-time inference at 5-10ms.

## License

This project is for research purposes. See [LICENSE](LICENSE) for details.
