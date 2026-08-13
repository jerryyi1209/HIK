#!/usr/bin/env python3
"""
精简 checkpoint — 移除训练状态，仅保留模型权重。

324 MB → ~108 MB (仅 model_state_dict + metrics)

用法:
    python strip_checkpoint.py                          # 精简 weights/best.pth
    python strip_checkpoint.py --input large.pth --output slim.pth
"""

import os, sys, argparse
import torch

def strip_checkpoint(input_path, output_path):
    print(f"Loading: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

    # 分析体积
    sizes = {}
    for k in ckpt:
        if isinstance(ckpt[k], dict):
            sizes[k] = sum(
                v.numel() * v.element_size()
                for v in ckpt[k].values()
                if isinstance(v, torch.Tensor)
            )
        elif isinstance(ckpt[k], torch.Tensor):
            sizes[k] = ckpt[k].numel() * ckpt[k].element_size()

    total_mb = sum(sizes.values()) / 1024 / 1024
    print(f"\n原始内容:")
    for k, s in sizes.items():
        print(f"  {k}: {s/1024/1024:.1f} MB")
    print(f"  总计: {total_mb:.1f} MB")

    # 精简：只保留模型权重和元信息
    slim = {
        'model_state_dict': ckpt['model_state_dict'],
        'epoch': ckpt.get('epoch', -1),
        'metrics': ckpt.get('metrics', {}),
        'in_channels': 9,
        'backbone': 'resnet50',
        'num_classes': 2,
        'use_refinement': True,
    }

    slim_mb = sum(
        v.numel() * v.element_size()
        for v in slim['model_state_dict'].values()
    ) / 1024 / 1024

    torch.save(slim, output_path)
    out_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"\n精简后: {out_mb:.1f} MB (权重 {slim_mb:.1f} MB)")
    print(f"节省: {os.path.getsize(input_path)/1024/1024 - out_mb:.1f} MB "
          f"({100*(1-out_mb/(os.path.getsize(input_path)/1024/1024)):.0f}%)")
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strip checkpoint for deployment')
    parser.add_argument('--input', type=str, default='weights/best.pth')
    parser.add_argument('--output', type=str, default='weights/best_deploy.pth')
    args = parser.parse_args()

    strip_checkpoint(args.input, args.output)
