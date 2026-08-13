#!/usr/bin/env python3
"""
RealSense D435 实时地毯分割推理

使用方法:
    python realtime_infer.py                          # 实时显示
    python realtime_infer.py --record ./output/       # 录制结果
    python realtime_infer.py --resolution 640 480     # 自定义分辨率
    python realtime_infer.py --no-display             # 无窗口（后台运行）

输入:
    D435 RGB + Depth 实时流

输出 (每帧):
    9ch tensor → SimpleModel → 二值分割掩码 → 叠加显示

性能 (Jetson Orin Nano, FP16 TensorRT):
    ~10-15 fps (含预处理)
"""

import os, sys, time, argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import deque

# 导入本地模型包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleModel, Preprocessor


# ═══════════════════════════════════════════════════════════════════════════
# D435 相机管理
# ═══════════════════════════════════════════════════════════════════════════

class D435Camera:
    """
    RealSense D435 相机封装。

    自动处理:
      - RGB/Depth 流对齐
      - 深度单位转换 (m)
      - 无效深度过滤
      - 自动曝光控制
    """

    def __init__(self, width=640, height=480, fps=30,
                 depth_width=640, depth_height=480):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_width = depth_width
        self.depth_height = depth_height

        import pyrealsense2 as rs
        self.rs = rs

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置 RGB 流
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )

        # 配置深度流
        self.config.enable_stream(
            rs.stream.depth, depth_width, depth_height, rs.format.z16, fps
        )

        # 启动
        self.profile = self.pipeline.start(self.config)

        # 对齐：深度 → RGB
        self.align = rs.align(rs.stream.color)

        # 深度传感器参数
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # m/unit

        # 预热 (前几帧曝光不稳定)
        print(f"[D435] 启动中，预热 {fps} 帧...")
        for _ in range(fps):
            self.pipeline.wait_for_frames()

        print(f"[D435] 就绪: RGB {width}x{height}@{fps}fps  "
              f"Depth {depth_width}x{depth_height} (scale={self.depth_scale:.4f}m)")

    def get_frames(self):
        """
        获取对齐后的 RGB-D 帧。

        Returns:
            rgb:   (H, W, 3) uint8 BGR
            depth: (H, W)    float32 meters
            timestamp: float 秒
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # RGB: BGR numpy
        rgb = np.asanyarray(color_frame.get_data())

        # Depth: uint16 → float32 meters
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth = depth_raw.astype(np.float32) * self.depth_scale

        ts = frames.get_timestamp() * 1e-3  # ms → s

        return rgb, depth, ts

    def stop(self):
        self.pipeline.stop()
        print("[D435] 已关闭")


# ═══════════════════════════════════════════════════════════════════════════
# 实时推理引擎
# ═══════════════════════════════════════════════════════════════════════════

class RealtimeSegmenter:
    """
    实时分割推理引擎。

    管线:
        D435帧 → resize(480,640) → 几何特征 → 9ch tensor → model.forward → mask → 叠加

    支持 PyTorch (默认) 和 ONNX Runtime 两种后端。
    """

    def __init__(self, checkpoint_path='weights/best_fp16.pth',
                 backbone='resnet50', in_channels=9,
                 image_size=(480, 640), device='cuda',
                 use_onnx=None, use_amp=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.use_onnx = use_onnx is not None

        if use_onnx:
            self._init_onnx(use_onnx)
        else:
            self._init_pytorch(checkpoint_path, backbone, in_channels)

        # 预处理器 (复用 deploy 包中的 Preprocessor)
        self.preprocessor = Preprocessor(
            target_size=image_size,
            device=str(self.device) if not use_onnx else 'cpu',
        )

        # 性能统计
        self.fps_buffer = deque(maxlen=60)
        self.last_infer_time = 0
        self.last_prep_time = 0

    def _init_pytorch(self, checkpoint_path, backbone, in_channels):
        print(f"[Model] PyTorch 后端: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)

        self.model = SimpleModel(
            backbone=backbone, pretrained=False,
            num_classes=2, use_refinement=True,
            in_channels=in_channels,
        ).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        src_miou = ckpt.get('metrics', {}).get('mIoU', '?')
        print(f"[Model] 加载完成: mIoU={src_miou}, device={self.device}, AMP={self.use_amp}")

        # GPU 预热
        if self.device.type == 'cuda':
            dummy = torch.randn(1, in_channels, *self.image_size, device=self.device)
            for _ in range(5):
                with torch.no_grad():
                    _ = self.model(dummy)
            torch.cuda.synchronize()
            print("[Model] GPU 预热完成")

    def _init_onnx(self, onnx_path):
        import onnxruntime as ort
        print(f"[Model] ONNX Runtime 后端: {onnx_path}")

        providers = []
        if self.device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Model] Providers: {providers}")

    @torch.no_grad()
    def predict(self, rgb, depth):
        """
        单帧推理。

        Args:
            rgb:   (H, W, 3) BGR uint8
            depth: (H, W) float32 meters

        Returns:
            mask:  (H_t, W_t) uint8 二值掩码 (255=地毯)
            prob:  (H_t, W_t) float32 概率图
        """
        t0 = time.perf_counter()

        # 预处理
        x = self.preprocessor.process(rgb, depth)
        self.last_prep_time = time.perf_counter() - t0

        # 推理
        t1 = time.perf_counter()

        if self.use_onnx:
            x_np = x.squeeze(0).cpu().numpy()[np.newaxis, ...]
            logits = self.session.run(None, {self.input_name: x_np})[0]
            prob = logits[0, 1]  # (H, W) carpet probability
        else:
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(x)
            prob = logits.float().softmax(dim=1)[0, 1].cpu().numpy()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        self.last_infer_time = time.perf_counter() - t1

        mask = (prob > 0.5).astype(np.uint8) * 255
        self.fps_buffer.append(1.0 / (time.perf_counter() - t0))

        return mask, prob

    @property
    def fps(self):
        return np.mean(self.fps_buffer) if self.fps_buffer else 0

    @property
    def latency_ms(self):
        return (self.last_prep_time + self.last_infer_time) * 1000


# ═══════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════

def draw_overlay(rgb, mask, alpha=0.45,
                 carpet_color=(50, 200, 50),
                 edge_color=(0, 255, 0)):
    """绿色半透明叠加地毯区域 + 轮廓线。"""
    H, W = rgb.shape[:2]
    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = rgb.copy()
    carpet = mask_resized > 127

    # 半透明填充
    overlay[carpet] = (
        overlay[carpet] * (1 - alpha) +
        np.array(carpet_color) * alpha
    ).astype(np.uint8)

    # 轮廓线
    edges = cv2.Canny(mask_resized, 50, 150)
    overlay[edges > 0] = edge_color

    return overlay


def draw_hud(frame, fps, latency_ms, carpet_pct, show_fps=True):
    """绘制信息 HUD (帧率、延迟、地毯占比)。"""
    h, w = frame.shape[:2]

    # 半透明信息栏
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    color_warn = (0, 200, 255)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                font, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {latency_ms:.0f}ms", (10, 52),
                font, 0.5, color, 1, cv2.LINE_AA)

    # 地毯占比 (有地毯时显示)
    if carpet_pct > 0.5:
        cv2.putText(frame, f"Carpet: {carpet_pct:.1f}%",
                    (w - 220, 38), font, 0.65, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Carpet Detected",
                    (w - 260, 38), font, 0.55, color_warn, 1, cv2.LINE_AA)

    return frame


def draw_depth_overlay(frame, depth, alpha=0.3):
    """小窗显示深度图 (右下角)。"""
    h, w = frame.shape[:2]
    dh, dw = 150, 200  # 小窗尺寸

    depth_small = cv2.resize(depth, (dw, dh))

    # 归一化 + 着色
    valid = depth_small > 0
    if valid.any():
        d_max = depth_small[valid].max()
        depth_small = np.clip(depth_small / max(d_max, 1e-6), 0, 1)
    depth_color = cv2.applyColorMap(
        (depth_small * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    depth_color[~(depth_small > 0)] = [30, 30, 30]

    # 叠加到右下角
    x0, y0 = w - dw - 10, h - dh - 10
    roi = frame[y0:y0 + dh, x0:x0 + dw]
    blended = cv2.addWeighted(roi, 1 - alpha, depth_color, alpha, 0)
    frame[y0:y0 + dh, x0:x0 + dw] = blended

    # 边框
    cv2.rectangle(frame, (x0, y0), (x0 + dw, y0 + dh), (180, 180, 180), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='RealSense D435 实时地毯分割',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模型
    parser.add_argument('--checkpoint', type=str, default='weights/best_fp16.pth')
    parser.add_argument('--onnx', type=str, default=None,
                        help='使用 ONNX Runtime 后端')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用 AMP 混合精度')

    # D435 相机
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480],
                        help='RGB 分辨率 (默认 640x480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='相机帧率')

    # 显示
    parser.add_argument('--no-display', action='store_true',
                        help='无窗口显示 (后台模式)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='全屏显示')
    parser.add_argument('--alpha', type=float, default=0.45,
                        help='分割重叠透明度')

    # 录制
    parser.add_argument('--record', type=str, default=None,
                        help='录制目录 (保存每帧的叠加结果)')
    parser.add_argument('--save-raw', action='store_true',
                        help='同时保存原始掩码')

    # 触发保存
    parser.add_argument('--save-trigger', type=str, default='s',
                        help='键盘按键保存当前帧 (默认 s)')

    args = parser.parse_args()

    # ── 创建推理引擎 ──
    segmenter = RealtimeSegmenter(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        device=args.device,
        use_onnx=args.onnx,
        use_amp=not args.no_amp,
    )

    # ── 录制目录 ──
    record_dir = None
    if args.record:
        record_dir = Path(args.record)
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / 'masks').mkdir(exist_ok=True)
        print(f"[Record] 录制目录: {record_dir}")

    # ── 启动 D435 ──
    try:
        camera = D435Camera(
            width=args.resolution[0],
            height=args.resolution[1],
            fps=args.fps,
        )
    except ImportError:
        print("错误: 未安装 pyrealsense2。请安装:")
        print("  pip install pyrealsense2")
        sys.exit(1)
    except RuntimeError as e:
        print(f"错误: 无法连接 D435 相机: {e}")
        print("请检查 USB 连接或尝试降低分辨率/帧率")
        sys.exit(1)

    # ── 显示窗口 ──
    window_name = 'CarpetSegNet v2 — D435 Real-time'
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

    # ── 统计 ──
    frame_idx = 0
    save_next = False
    t_start = time.perf_counter()

    print(f"\n{'='*55}")
    print(f"  实时分割已启动")
    print(f"  分辨率: {args.resolution[0]}x{args.resolution[1]} @ {args.fps}fps")
    print(f"  按 'q' 退出 | 按 '{args.save_trigger}' 保存当前帧")
    if args.record:
        print(f"  自动录制到: {record_dir}")
    print(f"{'='*55}\n")

    try:
        while True:
            # ── 获取帧 ──
            rgb, depth, ts = camera.get_frames()
            if rgb is None:
                continue

            frame_idx += 1

            # ── 推理 ──
            mask, prob = segmenter.predict(rgb, depth)

            # ── 可视化 ──
            display = draw_overlay(rgb, mask, alpha=args.alpha)
            display = draw_hud(
                display, segmenter.fps, segmenter.latency_ms,
                (mask > 127).mean() * 100
            )
            display = draw_depth_overlay(display, depth)

            # ── 录制 ──
            if args.record or save_next:
                out_dir = record_dir if record_dir else Path('./saved_frames')
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / 'masks').mkdir(exist_ok=True)

                fname = f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(out_dir / fname), display)

                if args.save_raw or save_next:
                    cv2.imwrite(str(out_dir / 'masks' / fname), mask)

                if save_next:
                    print(f"[Save] {fname}")
                    save_next = False

            # ── 显示 ──
            if not args.no_display:
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(args.save_trigger):
                    save_next = True

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\n总计: {frame_idx} 帧 in {elapsed:.1f}s ({avg_fps:.1f} fps avg)")
        if record_dir:
            print(f"录制文件: {record_dir}")


if __name__ == '__main__':
    main()
