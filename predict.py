# 推理入口脚本 - 单张/批量图片推理
import argparse
import torch
import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DetectionModel, predict


def parse_args():
    parser = argparse.ArgumentParser(description="目标检测推理")
    parser.add_argument("--model", required=True, help="模型文件路径 (.pth)")
    parser.add_argument("--input", required=True, help="输入图片/文件夹路径")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640], help="输入尺寸")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="计算设备")
    parser.add_argument("--save-vis", default=True, action="store_true", help="保存可视化结果")
    parser.add_argument("--save-json", default=True, action="store_true", help="保存 JSON 结果")
    parser.add_argument("--classes", type=str, default=None, help="类别名称文件 (每行一个)")
    return parser.parse_args()


def load_classes(classes_file):
    """加载类别名称"""
    if not classes_file or not os.path.exists(classes_file):
        return None
    with open(classes_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def draw_detections(image, detections, classes=None):
    """在图片上绘制检测框"""
    vis_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        score = det["score"]
        class_id = det["class_id"]
        
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{classes[class_id] if classes and class_id < len(classes) else f'class_{class_id}'} {score:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis_img


def process_image(model, image_path, output_dir, args, classes=None):
    """处理单张图片"""
    print(f"处理: {image_path}")
    
    detections = predict(
        model, 
        image_path, 
        device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=tuple(args.img_size)
    )
    
    print(f"  检测到 {len(detections)} 个目标")
    
    # 保存 JSON
    if args.save_json:
        json_path = Path(output_dir) / (Path(image_path).stem + ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "image": str(image_path),
                "detections": detections
            }, f, ensure_ascii=False, indent=2)
    
    # 可视化
    if args.save_vis:
        img = cv2.imread(image_path)
        if img is not None:
            vis_img = draw_detections(img, detections, classes)
            vis_path = Path(output_dir) / (Path(image_path).stem + "_vis.jpg")
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis_img)
            print(f"  可视化保存至: {vis_path}")
    
    return detections


def main():
    args = parse_args()
    
    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    checkpoint = torch.load(args.model, map_location=device)
    num_classes = checkpoint.get('num_classes', 1)
    backbone = checkpoint.get('backbone', 'resnet18')
    
    model = DetectionModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"模型加载成功: {num_classes} 类别, backbone={backbone}")
    
    # 加载类别名称
    classes = load_classes(args.classes)
    if classes:
        print(f"加载类别: {classes}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    if input_path.is_file():
        # 单张图片
        process_image(model, str(input_path), args.output, args, classes)
    elif input_path.is_dir():
        # 批量处理文件夹
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_exts:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"未找到图片文件: {args.input}")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始批量推理...")
        for img_path in sorted(image_files):
            try:
                process_image(model, str(img_path), args.output, args, classes)
            except Exception as e:
                print(f"处理 {img_path} 失败: {e}")
        print(f"\n批量推理完成，结果保存在: {args.output}")
    else:
        print(f"错误: 输入路径既不是文件也不是目录: {args.input}")


if __name__ == "__main__":
    main()
