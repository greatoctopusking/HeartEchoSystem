#!/usr/bin/env python
"""
命令行接口
使用方法:
    python -m inference --checkpoint xxx.pth --input image.jpg
    python -m inference --checkpoint xxx.pth --input-dir ./images/
    python -m inference --config inference_config.yaml
"""
import argparse
import os
import sys
import torch

from .predict import predict_single, predict_directory, load_model
from .transforms import get_val_transforms
from .constants import NUM_CLASSES, DEFAULT_IMAGE_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description='心脏超声切面分类模型推理',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='predict',
        help='命令 (predict)'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=False,
        help='模型权重文件路径'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='单张图像路径'
    )
    
    parser.add_argument(
        '--input-dir', '-d',
        type=str,
        help='图像目录路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='inference_results.json',
        help='输出JSON文件路径'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='批大小'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        default=NUM_CLASSES,
        help='分类类别数'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help='图像尺寸'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='推理设备 (cuda/cpu)'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='递归搜索子目录'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # 先加载配置文件（如果提供）
    if args.config:
        config = load_config(args.config)
        args.checkpoint = config.get('checkpoint', args.checkpoint)
        args.input = config.get('input', args.input)
        args.input_dir = config.get('input_dir', args.input_dir)
        args.output = config.get('output', args.output)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.num_classes = config.get('num_classes', args.num_classes)
        args.device = config.get('device', args.device)
        args.recursive = config.get('recursive', args.recursive)
    
    # 优先级：命令行参数 > 配置文件 > 默认值
    # 如果 checkpoint 仍为空，报错
    if not args.checkpoint:
        print("Error: --checkpoint is required")
        print("  Use: --checkpoint <path> or --config <yaml>")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input-dir is required")
        sys.exit(1)
    
    print(f"Using device: {args.device}")
    print(f"Loading model from {args.checkpoint}...")
    
    transform = get_val_transforms(args.image_size)
    model = load_model(args.checkpoint, num_classes=args.num_classes, device=args.device)
    
    print("Model loaded successfully")
    
    if args.input:
        print(f"\nPredicting single image: {args.input}")
        result = predict_single(model, args.input, transform=transform, device=args.device)
        
        print(f"\nResult:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence'] * 100:.2f}%")
        print(f"\nAll probabilities:")
        for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
            print(f"  {cls}: {prob * 100:.2f}%")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to {args.output}")
    
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Directory not found: {args.input_dir}")
            sys.exit(1)
        
        print(f"\nPredicting images in directory: {args.input_dir}")
        output = predict_directory(
            model,
            args.input_dir,
            output_path=args.output,
            transform=transform,
            device=args.device,
            batch_size=args.batch_size,
            recursive=args.recursive
        )
        
        print(f"\nSummary:")
        print(f"  Total images: {output['total_images']}")
        print(f"  Inference time: {output['inference_time_seconds']}s")
        print(f"\nClass distribution:")
        for cls, count in output['class_distribution'].items():
            print(f"  {cls}: {count}")
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()