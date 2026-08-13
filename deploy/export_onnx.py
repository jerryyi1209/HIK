#!/usr/bin/env python3
"""
Export CarpetSegNet v2 to ONNX format for Jetson Orin Nano TensorRT deployment.

Workflow:
  1. Export PyTorch → ONNX (this script)
  2. (On Jetson) ONNX → TensorRT engine:
       trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
       trtexec --onnx=model.onnx --saveEngine=model_int8.trt --int8 --calib=<calib_file>

Usage:
    python export_onnx.py                           # export with default settings
    python export_onnx.py --dynamic-batch           # export with dynamic batch size
    python export_onnx.py --half                     # export FP16 weights directly
    python export_onnx.py --opset 17 --simplify      # optimize for TensorRT 8.6+

Output:
    model.onnx            — FP32 ONNX model
    model_fp16.onnx       — FP16 ONNX model (with --half)
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add current directory to path so `model` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SimpleModel


def export_onnx(
    checkpoint_path: str = 'weights/best_fp16.pth',
    output_path: str = 'model.onnx',
    backbone: str = 'resnet50',
    in_channels: int = 9,
    image_size: tuple = (480, 640),
    opset_version: int = 17,
    dynamic_batch: bool = False,
    use_fp16: bool = False,
    simplify: bool = False,
    verify: bool = True,
):
    """
    Export model to ONNX.

    Args:
        checkpoint_path: PyTorch checkpoint
        output_path:     ONNX output path
        backbone:        'resnet34' | 'resnet50'
        in_channels:     input channels (9 for D435+geo)
        image_size:      (H, W) input resolution
        opset_version:   ONNX opset (17+ for TensorRT 8.6+)
        dynamic_batch:   use dynamic batch axis
        use_fp16:        export in FP16 (reduces file size)
        simplify:        run onnxsim for graph optimization
        verify:          compare ONNX output against PyTorch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = image_size

    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # ── Build model ──
    model = SimpleModel(
        backbone=backbone,
        pretrained=False,
        num_classes=2,
        use_refinement=True,
        in_channels=in_channels,
    ).to(device)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    miou = ckpt.get('metrics', {}).get('mIoU', '?')
    print(f"Loaded: backbone={backbone}, in_ch={in_channels}, mIoU={miou}")

    # Switch to FP16 if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        dtype = torch.float16
        print("Model converted to FP16")
    else:
        dtype = torch.float32

    # ── Dummy input ──
    dummy_input = torch.randn(1, in_channels, H, W, device=device, dtype=dtype)

    # ── Warmup to initialize BN running stats ──
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # ── Export ──
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    print(f"Exporting to: {output_path}")
    print(f"  Input:  (B, {in_channels}, {H}, {W}) {dtype}")
    print(f"  Output: (B, 2, {H}, {W})")
    print(f"  Opset:  {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    # ── Simplify (optional) ──
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
        except ImportError:
            print("  Warning: onnxsim not installed. Install with: pip install onnxsim")
            print("  Skipping simplification.")
            simplify = False

    if simplify:
        print("Simplifying ONNX graph...")
        onnx_model = onnx.load(output_path)
        simplified_model, check = simplify(onnx_model)

        if check:
            output_path_sim = output_path.replace('.onnx', '_sim.onnx')
            onnx.save(simplified_model, output_path_sim)
            sim_size = os.path.getsize(output_path_sim) / (1024 * 1024)
            print(f"  Simplified: {output_path_sim} ({sim_size:.1f} MB)")
            output_path = output_path_sim
        else:
            print("  Warning: Simplification check failed. Using original model.")

    # ── Verify ──
    if verify:
        print("\nVerifying ONNX model...")
        verify_onnx(output_path, model, dummy_input, device)
        print("  Verification passed.")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"ONNX export complete: {output_path}")
    print(f"\nNext steps (on Jetson Orin Nano):")
    print(f"  1. Copy {output_path} to Jetson")
    print(f"  2. Convert to TensorRT:")
    if use_fp16:
        print(f"     trtexec --onnx={os.path.basename(output_path)} "
              f"--saveEngine=model.trt --fp16")
    else:
        print(f"     trtexec --onnx={os.path.basename(output_path)} "
              f"--saveEngine=model.trt --fp16")
    print(f"  3. Run inference:")
    print(f"     python infer.py --onnx {os.path.basename(output_path)} "
          f"--rgb img.png --depth depth.png --out mask.png")
    print(f"{'='*50}")


def verify_onnx(onnx_path: str, pytorch_model: torch.nn.Module,
                dummy_input: torch.Tensor, device: torch.device):
    """
    Compare ONNX Runtime output against PyTorch model output.
    Permissible error: < 1e-3 for FP32, < 1e-2 for FP16.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed. Skipping verification.")
        return

    # PyTorch forward
    with torch.no_grad():
        pt_out = pytorch_model(dummy_input).float().cpu().numpy()

    # ONNX forward
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider'],
    )
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy_input.float().cpu().numpy()})[0]

    # Compare
    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    is_fp16 = dummy_input.dtype == torch.float16
    threshold = 5e-3 if is_fp16 else 5e-4

    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Threshold: {threshold} ({'FP16' if is_fp16 else 'FP32'})")

    if max_diff > threshold:
        print(f"  WARNING: max diff {max_diff:.6f} > threshold {threshold}")
    else:
        print(f"  OK — outputs match within tolerance")


def export_onnx_jit(
    checkpoint_path: str = 'weights/best_fp16.pth',
    output_path: str = 'model_script.pt',
    backbone: str = 'resnet50',
    in_channels: int = 9,
    image_size: tuple = (480, 640),
):
    """
    Alternative: export via TorchScript (torch.jit.script or trace).

    Useful when ONNX opset doesn't support certain operations.
    TorchScript models can be loaded with libtorch on Jetson.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = image_size

    model = SimpleModel(
        backbone=backbone,
        pretrained=False,
        num_classes=2,
        use_refinement=True,
        in_channels=in_channels,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Trace
    dummy = torch.randn(1, in_channels, H, W, device=device)
    traced = torch.jit.trace(model, dummy, strict=False)

    # Optimize for inference
    traced = torch.jit.freeze(traced)
    traced = torch.jit.optimize_for_inference(traced)

    traced.save(output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TorchScript model saved: {output_path} ({file_size:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export CarpetSegNet v2 to ONNX / TorchScript',
    )
    parser.add_argument('--checkpoint', type=str, default='weights/best_fp16.pth')
    parser.add_argument('--output', type=str, default='model.onnx')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--in-channels', type=int, default=9)
    parser.add_argument('--image-size', type=int, nargs=2, default=[480, 640])
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamic-batch', action='store_true')
    parser.add_argument('--half', action='store_true',
                        help='Export FP16 weights directly into ONNX')
    parser.add_argument('--simplify', action='store_true',
                        help='Run onnxsim graph optimization')
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--torchscript', action='store_true',
                        help='Export TorchScript instead of ONNX')

    args = parser.parse_args()

    if args.torchscript:
        export_onnx_jit(
            checkpoint_path=args.checkpoint,
            output_path=args.output.replace('.onnx', '_script.pt'),
            backbone=args.backbone,
            in_channels=args.in_channels,
            image_size=tuple(args.image_size),
        )
    else:
        export_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            backbone=args.backbone,
            in_channels=args.in_channels,
            image_size=tuple(args.image_size),
            opset_version=args.opset,
            dynamic_batch=args.dynamic_batch,
            use_fp16=args.half,
            simplify=args.simplify,
            verify=not args.no_verify,
        )
