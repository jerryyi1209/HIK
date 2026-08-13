# CarpetSegNet v2 — D435 Deployment

Jetson Orin Nano (Ubuntu 22.04) inference deployment for carpet segmentation.

## Model

- **Architecture**: ResNet50 encoder + U-Net decoder + RefinementHead
- **Input**: 9 channels — RGB(3) + Depth(1) + Surface Normals(3) + Depth Edge(1) + Curvature(1)
- **Output**: Binary carpet mask (255=carpet, 0=background)
- **Input size**: 480×640
- **Params**: ~26M
- **Best mIoU**: 0.8272 (August 7, 2026 — NYU augmented, label smoothing + weight decay)
- **Checkpoint**: `weights/best_fp16.pth` (FP16 — 54 MB, half-precision weights)

## Directory Structure

```
deploy/
├── model/
│   ├── __init__.py              # Package exports
│   ├── simple_model.py          # Main model (encoder + decoder)
│   ├── simple_encoder.py        # ResNet encoder (9ch input)
│   ├── simple_decoder.py        # U-Net decoder + RefinementHead
│   └── preprocess.py            # Depth normalization + geometric features
├── weights/
│   └── best_fp16.pth            # Best checkpoint (mIoU=0.8272, FP16)
├── infer.py                     # Inference script (PyTorch / ONNX)
├── realtime_infer.py            # Real-time D435 camera inference
├── export_onnx.py               # ONNX / TorchScript export
├── requirements.txt             # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# On Jetson Orin Nano (PyTorch is pre-installed via JetPack)
pip install -r requirements.txt

# For ONNX inference (optional)
pip install onnxruntime-gpu
```

### 2. Run Inference

```bash
# Single image (PyTorch)
python infer.py \
    --rgb test_rgb.png \
    --depth test_depth.png \
    --out mask.png

# Batch directory
python infer.py \
    --rgb-dir ./test_rgb/ \
    --depth-dir ./test_depth/ \
    --out-dir ./pred_masks/ \
    --vis-dir ./vis_overlay/

# Benchmark FPS
python infer.py \
    --rgb test_rgb.png --depth test_depth.png \
    --benchmark
```

### 3. Export to ONNX (for TensorRT)

```bash
# On workstation or Jetson
python export_onnx.py \
    --checkpoint weights/best_fp16.pth \
    --output model.onnx \
    --opset 17 \
    --simplify

# FP16 export (smaller file, faster on TensorRT)
python export_onnx.py \
    --checkpoint weights/best_fp16.pth \
    --output model_fp16.onnx \
    --half --simplify
```

### 4. Convert to TensorRT Engine (on Jetson)

```bash
# FP16 TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model.trt \
    --fp16 \
    --minShapes=input:1x9x480x640 \
    --optShapes=input:1x9x480x640 \
    --maxShapes=input:8x9x480x640

# ONNX Runtime with TensorRT backend
python infer.py \
    --onnx model.onnx \
    --rgb test_rgb.png --depth test_depth.png \
    --out mask.png
```

## Input Format

| Channel | Content | Source | Normalization |
|---------|---------|--------|---------------|
| 0-2 | RGB | Camera image | ImageNet mean/std |
| 3 | Depth | D435 depth frame | Per-frame min/max |
| 4-6 | Surface normals | Computed from depth | Unit vectors |
| 7 | Depth edge | Sobel on depth | Per-frame min/max |
| 8 | Curvature | Laplacian on depth | Per-frame min/max |

**Depth input**: 16-bit PNG images with values in millimeters.
Convert from D435 depth frame (meters → mm × 1000 → uint16).

## Performance Notes

- **PyTorch FP32**: ~15-25 ms/inference on Jetson Orin Nano
- **PyTorch FP16 (AMP)**: ~10-15 ms/inference
- **TensorRT FP16**: ~5-10 ms/inference (after engine build)
- **Preprocessing**: ~3-5 ms (geometric features on GPU with CUDA)

### 5. Real-time D435 Inference

```bash
# Real-time display
python realtime_infer.py

# Record results
python realtime_infer.py --record ./recorded/

# Lower resolution for higher FPS
python realtime_infer.py --resolution 640 480 --fps 30

# With ONNX backend
python realtime_infer.py --onnx model.onnx

# No display (headless/background)
python realtime_infer.py --no-display --record ./output/

# Controls:
#   'q' — quit
#   's' — save current frame to disk

# Custom save trigger key
python realtime_infer.py --save-trigger space  # use spacebar to save
```

## ONNX → TensorRT Pipeline (recommended)

For maximum throughput on Jetson:

```
PyTorch checkpoint ──[export_onnx.py]──→ ONNX ──[trtexec]──→ TensorRT engine
                                                                    │
                                                    infer.py --trt model.trt ...
```

TensorRT engine inference support (via pycuda/tensorrt Python bindings)
is planned. Currently use ONNX Runtime with CUDA backend as the intermediate
solution.
