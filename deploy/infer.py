#!/usr/bin/env python3
"""
CarpetSegNet v2 — D435 Inference Script for Jetson Orin Nano.

Supports three modes:
  1. PyTorch inference (default)
  2. ONNX Runtime inference (--onnx)
  3. TensorRT inference (--trt)

Input:  RGB image (PNG/JPG) + Depth image (16-bit PNG, mm)
Output: Binary carpet mask (PNG)

Usage:
    # Single image
    python infer.py --rgb img_0001_rgb.png --depth img_0001_depth.png --out mask.png

    # Batch directory
    python infer.py --rgb-dir ./test_rgb/ --depth-dir ./test_depth/ --out-dir ./pred/

    # With ONNX model
    python infer.py --onnx model.onnx --rgb img.png --depth depth.png --out mask.png

    # With TensorRT engine (Jetson optimized)
    python infer.py --trt model.trt --rgb img.png --depth depth.png --out mask.png
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch

from model import SimpleModel, Preprocessor


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Inference
# ═══════════════════════════════════════════════════════════════════════════

class PyTorchInference:
    """PyTorch-based inference pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = 'resnet50',
        in_channels: int = 9,
        image_size: tuple = (480, 640),
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device.type == 'cuda'

        print(f"[PyTorch] Loading checkpoint: {checkpoint_path}")
        print(f"[PyTorch] Device: {self.device}, AMP: {self.use_amp}")

        # Build model
        self.model = SimpleModel(
            backbone=backbone,
            pretrained=False,
            num_classes=2,
            use_refinement=True,
            in_channels=in_channels,
        ).to(self.device)

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)

        # Handle key mismatch
        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(state.keys())
        missing = model_keys - ckpt_keys
        if missing:
            print(f"  Warning: {len(missing)} keys missing (will be random init):")
            for k in sorted(missing):
                print(f"    - {k}")

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Log checkpoint info
        epoch = ckpt.get('epoch', '?')
        miou = ckpt.get('metrics', {}).get('mIoU', '?')
        print(f"[PyTorch] Source: epoch={epoch}, mIoU={miou}")

        # Preprocessor
        self.preprocessor = Preprocessor(
            target_size=image_size,
            device=str(self.device),
        )

        # Warmup
        print("[PyTorch] Warming up...")
        dummy = torch.randn(1, in_channels, *image_size, device=self.device)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy)
        print("[PyTorch] Ready.")

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Run inference on a single RGB-D pair.

        Args:
            rgb:   (H, W, 3) uint8
            depth: (H, W) float32, meters
        Returns:
            mask: (H_t, W_t) uint8 — 255=carpet, 0=background
        """
        x = self.preprocessor.process(rgb, depth)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            logits = self.model(x)

        prob = logits.float().softmax(dim=1)[0, 1]  # carpet channel
        mask = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255

        return mask

    @torch.no_grad()
    def predict_proba(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Return carpet probability map (float, 0-1)."""
        x = self.preprocessor.process(rgb, depth)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            logits = self.model(x)

        return logits.float().softmax(dim=1)[0, 1].cpu().numpy()

    def benchmark(self, rgb: np.ndarray, depth: np.ndarray, n_warmup: int = 10, n_runs: int = 100):
        """Measure inference FPS."""
        x = self.preprocessor.process(rgb, depth)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = self.model(x)

        # Sync and time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model(x)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        fps = n_runs / elapsed
        latency_ms = (elapsed / n_runs) * 1000
        print(f"[Benchmark] {n_runs} runs: {elapsed:.3f}s total, "
              f"{latency_ms:.1f}ms/inference, {fps:.1f} FPS")


# ═══════════════════════════════════════════════════════════════════════════
# ONNX Runtime Inference
# ═══════════════════════════════════════════════════════════════════════════

class ONNXInference:
    """ONNX Runtime inference (CPU or CUDA)."""

    def __init__(
        self,
        onnx_path: str,
        image_size: tuple = (480, 640),
        device: str = 'cuda',
    ):
        import onnxruntime as ort

        self.image_size = image_size

        # Select execution provider
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        print(f"[ONNX] Loading: {onnx_path}")
        print(f"[ONNX] Providers: {providers}")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.device = 'cuda' if device == 'cuda' else 'cpu'

        self.preprocessor = Preprocessor(
            target_size=image_size,
            device='cpu',  # ONNX expects numpy/cpu input
        )

        print(f"[ONNX] Input: {self.input_name}, "
              f"shape={self.session.get_inputs()[0].shape}")
        print(f"[ONNX] Ready.")

    def predict(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Run inference. Returns (H, W) uint8 mask."""
        x = self.preprocessor.process(rgb, depth)  # (1, 9, H, W) on cpu
        x_np = x.squeeze(0).cpu().numpy()  # (9, H, W)

        # ONNX expects NCHW
        x_np = x_np[np.newaxis, ...]  # (1, 9, H, W)

        logits = self.session.run(None, {self.input_name: x_np})[0]
        # logits: (1, 2, H, W)

        prob = logits[0, 1]  # carpet channel
        mask = (prob > 0.5).astype(np.uint8) * 255
        return mask

    def benchmark(self, rgb: np.ndarray, depth: np.ndarray,
                  n_warmup: int = 10, n_runs: int = 100):
        """Measure inference FPS."""
        x = self.preprocessor.process(rgb, depth)
        x_np = x.squeeze(0).cpu().numpy()[np.newaxis, ...]

        for _ in range(n_warmup):
            _ = self.session.run(None, {self.input_name: x_np})

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.session.run(None, {self.input_name: x_np})
        elapsed = time.perf_counter() - t0

        fps = n_runs / elapsed
        latency_ms = (elapsed / n_runs) * 1000
        print(f"[Benchmark] {n_runs} runs: {elapsed:.3f}s total, "
              f"{latency_ms:.1f}ms/inference, {fps:.1f} FPS")


# ═══════════════════════════════════════════════════════════════════════════
# Batch Processing Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_depth_png(path: str) -> np.ndarray:
    """Load 16-bit depth PNG (mm), convert to float32 meters."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    if d.dtype == np.uint16:
        d = d.astype(np.float32) / 1000.0  # mm → m
    return d


def load_rgb(path: str) -> np.ndarray:
    """Load RGB image as uint8."""
    rgb = cv2.imread(path)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4,
                 color: tuple = (0, 255, 0)) -> np.ndarray:
    """Overlay binary mask on RGB image for visualization."""
    overlay = rgb.copy()
    overlay[mask > 127] = (
        overlay[mask > 127] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    return overlay


def process_directory(infer_engine, rgb_dir: str, depth_dir: str,
                      out_dir: str, vis_dir: str = None,
                      ext: str = '.png'):
    """Batch process all images in a directory."""
    os.makedirs(out_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(ext)])
    print(f"\nProcessing {len(rgb_files)} images...")

    total_time = 0.0
    for i, fname in enumerate(rgb_files):
        rgb_path = os.path.join(rgb_dir, fname)
        depth_path = os.path.join(depth_dir, fname)

        rgb = load_rgb(rgb_path)
        depth = load_depth_png(depth_path)

        t0 = time.perf_counter()
        mask = infer_engine.predict(rgb, depth)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Save mask
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, mask)

        # Save visualization
        if vis_dir:
            vis = overlay_mask(rgb, mask)
            vis_path = os.path.join(vis_dir, fname)
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        carpet_pct = (mask > 127).mean() * 100
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{len(rgb_files)}] "
                  f"{elapsed*1000:.1f}ms | carpet: {carpet_pct:.1f}% | {fname}")

    avg_latency = (total_time / len(rgb_files)) * 1000
    print(f"\nDone. {len(rgb_files)} images in {total_time:.1f}s "
          f"({avg_latency:.1f}ms avg)")
    print(f"Masks: {out_dir}")
    if vis_dir:
        print(f"Visualizations: {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='CarpetSegNet v2 — D435 Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / Output ──
    parser.add_argument('--rgb', type=str, default=None,
                        help='RGB image path (single-image mode)')
    parser.add_argument('--depth', type=str, default=None,
                        help='Depth image path (16-bit PNG, mm)')
    parser.add_argument('--out', type=str, default='mask.png',
                        help='Output mask path')
    parser.add_argument('--rgb-dir', type=str, default=None,
                        help='RGB directory (batch mode)')
    parser.add_argument('--depth-dir', type=str, default=None,
                        help='Depth directory (batch mode)')
    parser.add_argument('--out-dir', type=str, default='./pred_masks',
                        help='Output directory (batch mode)')
    parser.add_argument('--vis-dir', type=str, default=None,
                        help='Visualization overlay directory')

    # ── Model ──
    parser.add_argument('--checkpoint', type=str, default='weights/best_fp16.pth',
                        help='PyTorch checkpoint path')
    parser.add_argument('--onnx', type=str, default=None,
                        help='ONNX model path (enables ONNX Runtime inference)')
    parser.add_argument('--trt', type=str, default=None,
                        help='TensorRT engine path (not yet implemented)')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--in-channels', type=int, default=9)
    parser.add_argument('--image-size', type=int, nargs=2, default=[480, 640])

    # ── Performance ──
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--benchmark', action='store_true',
                        help='Run FPS benchmark')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP (PyTorch inference only)')

    # ── Output options ──
    parser.add_argument('--prob', action='store_true',
                        help='Output probability map instead of binary mask')
    parser.add_argument('--ext', type=str, default='.png',
                        help='Image file extension for batch mode')

    args = parser.parse_args()

    # ── Create inference engine ──
    if args.onnx:
        engine = ONNXInference(
            onnx_path=args.onnx,
            image_size=tuple(args.image_size),
            device=args.device,
        )
    elif args.trt:
        raise NotImplementedError(
            "TensorRT inference not yet implemented. "
            "Export to ONNX first: python export_onnx.py"
        )
    else:
        engine = PyTorchInference(
            checkpoint_path=args.checkpoint,
            backbone=args.backbone,
            in_channels=args.in_channels,
            image_size=tuple(args.image_size),
            device=args.device,
            use_amp=not args.no_amp,
        )

    # ── Batch mode ──
    if args.rgb_dir and args.depth_dir:
        process_directory(
            infer_engine=engine,
            rgb_dir=args.rgb_dir,
            depth_dir=args.depth_dir,
            out_dir=args.out_dir,
            vis_dir=args.vis_dir,
            ext=args.ext,
        )
        return

    # ── Single-image mode ──
    if not args.rgb or not args.depth:
        parser.error("Specify --rgb/--depth (single) or --rgb-dir/--depth-dir (batch)")

    rgb = load_rgb(args.rgb)
    depth = load_depth_png(args.depth)

    # Benchmark
    if args.benchmark:
        engine.benchmark(rgb, depth)

    # Inference
    t0 = time.perf_counter()
    if args.prob:
        result = engine.predict_proba(rgb, depth)
        # Save as 16-bit PNG for precision
        result_u16 = (result * 65535).astype(np.uint16)
        cv2.imwrite(args.out, result_u16)
    else:
        result = engine.predict(rgb, depth)
        cv2.imwrite(args.out, result)

    elapsed = (time.perf_counter() - t0) * 1000
    carpet_pct = (result > 127).mean() * 100 if not args.prob else 0

    print(f"\nInference: {elapsed:.1f}ms")
    if not args.prob:
        print(f"Carpet:   {carpet_pct:.1f}%")
    print(f"Saved:    {args.out}")


if __name__ == '__main__':
    main()
