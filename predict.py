import torch
import cv2
import numpy as np
import os, sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DetectionModel, predict

def get_env(key, default):
    return os.environ.get(key, default)

def get_env_bool(key, default):
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")

def get_env_int(key, default):
    return int(os.environ.get(key, str(default)))

def get_env_float(key, default):
    return float(os.environ.get(key, str(default)))

def get_env_tuple(key, default):
    val = os.environ.get(key)
    if val:
        return tuple(map(int, val.split()))
    return default

def load_classes(classes_file):
    if not classes_file or not os.path.exists(classes_file):
        return None
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def draw_detections(image, detections, classes=None):
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

def process_image(model, image_path, output_dir, conf, iou, img_size, device, save_vis, save_json, classes):
    print(f"处理: {image_path}")
    detections = predict(
        model, image_path, device=device, conf_thresh=conf, iou_thresh=iou, img_size=img_size
    )
    print(f"  检测到 {len(detections)} 个目标")
    if save_json:
        json_path = Path(output_dir) / (Path(image_path).stem + ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"image": str(image_path), "detections": detections}, f, ensure_ascii=False, indent=2)
    if save_vis:
        img = cv2.imread(image_path)
        if img is not None:
            vis_img = draw_detections(img, detections, classes)
            vis_path = Path(output_dir) / (Path(image_path).stem + "_vis.jpg")
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis_img)
            print(f"  可视化保存至: {vis_path}")
    return detections

def main():
    model_path = get_env("MODEL_PATH", "")
    input_path = get_env("INPUT_PATH", "")
    output_dir = get_env("OUTPUT_DIR", "output")
    conf = get_env_float("CONF", 0.5)
    iou = get_env_float("IOU", 0.5)
    img_size = get_env_tuple("PRED_IMG_SIZE", (640, 640))
    device = get_env("PRED_DEVICE", "auto")
    save_vis = get_env_bool("SAVE_VIS", True)
    save_json = get_env_bool("SAVE_JSON", True)
    classes_file = get_env("CLASSES", None)
    
    if not model_path or not input_path:
        print("错误: 缺少 MODEL_PATH 或 INPUT_PATH 环境变量")
        return
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 设置本地缓存
    os.environ.setdefault("TORCH_HOME", os.path.join(os.getcwd(), "torch_cache"))
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 1)
    backbone = checkpoint.get("backbone", "resnet50")
    model_type = checkpoint.get("model_type", "fasterrcnn")
    
    # 创建模型结构（pretrained=False，权重从 checkpoint 加载）
    model = DetectionModel(num_classes=num_classes, backbone=backbone, pretrained=False, device=device, model_type=model_type)
    # 加载内部模型的 state_dict
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"模型加载成功: {num_classes} 类别, backbone={backbone}, type={model_type}")
    
    classes = load_classes(classes_file) if classes_file else None
    if classes:
        print(f"加载类别: {classes}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return
    
    if input_path_obj.is_file():
        process_image(model, str(input_path_obj), output_dir, conf, iou, img_size, device, True, True, classes)
    elif input_path_obj.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []
        for ext in image_exts:
            image_files.extend(input_path_obj.glob(f"*{ext}"))
            image_files.extend(input_path_obj.glob(f"*{ext.upper()}"))
        if not image_files:
            print(f"未找到图片文件: {input_path}")
            return
        print(f"找到 {len(image_files)} 张图片，开始批量推理...")
        for img_path in sorted(image_files):
            try:
                process_image(model, str(img_path), output_dir, conf, iou, img_size, device, True, True, None)
            except Exception as e:
                print(f"处理 {img_path} 失败: {e}")
        print(f"批量推理完成，结果保存在: {output_dir}")
    else:
        print(f"错误: 输入路径既不是文件也不是目录: {input_path}")

if __name__ == "__main__":
    main()
