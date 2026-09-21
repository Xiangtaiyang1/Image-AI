import torch, torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetHead
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2, numpy as np, json, math, os
from pathlib import Path
from typing import List, Dict, Tuple

STRIDE = 32  # 兼容性常量

def _letterbox(image, img_size):
    w, h = img_size
    ih, iw = image.shape[:2]
    scale = min(h/ih, w/iw)
    nw, nh = max(1,int(iw*scale)), max(1,int(ih*scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


class DetectionDataset(Dataset):
    """兼容原格式的数据集，输出 torchvision 格式的 target"""
    def __init__(self, marked_dir, unmarked_dir, img_size=(640,640)):
        self.marked_dir = Path(marked_dir)
        self.unmarked_dir = Path(unmarked_dir)
        self.img_size = tuple(img_size)
        self.samples = []
        exts = (".jpg",".jpeg",".png",".bmp",".tiff",".tif")
        for jf in sorted(self.marked_dir.glob("*.json")):
            try:
                ann = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            img_p = self.unmarked_dir / ann.get('image_name', '')
            if not img_p.exists():
                for e in exts:
                    cand = self.unmarked_dir / (jf.stem + e)
                    if cand.exists():
                        img_p = cand
                        break
            if not img_p.exists() or not img_p.is_file():
                continue
            boxes = []
            for a in ann.get('annotations', []):
                if len(a.get('bbox_abs', []) or []) == 4:
                    x1,y1,x2,y2 = map(int, a['bbox_abs'])
                elif len(a.get('bbox', []) or []) == 4:
                    iw0 = max(1, ann.get('image_width', 1)); ih0 = max(1, ann.get('image_height',1))
                    cx,cy,bw,bh = a['bbox']
                    x1 = int((cx-bw/2)*iw0); y1 = int((cy-bh/2)*ih0)
                    x2 = int((cx+bw/2)*iw0); y2 = int((cy+bh/2)*ih0)
                else:
                    continue
                if x2-x1 < 2 or y2-y1 < 2:
                    continue
                boxes.append([x1,y1,x2,y2])
            if boxes:
                self.samples.append((img_p, boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, boxes = self.samples[idx]
        img = cv2.imread(str(img_p))
        if img is None:
            img = np.zeros((self.img_size[1], self.img_size[0],3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, scale = _letterbox(img, self.img_size)
        # 转为 tensor [0,1]
        img_t = torch.from_numpy(canvas).permute(2,0,1).float() / 255.0
        # 将 boxes 缩放到 letterbox 坐标系
        scaled_boxes = []
        for (x1,y1,x2,y2) in boxes:
            scaled_boxes.append([x1*scale, y1*scale, x2*scale, y2*scale])
        target = {
            "boxes": torch.tensor(scaled_boxes, dtype=torch.float32),  # [N,4] xyxy
            "labels": torch.ones(len(scaled_boxes), dtype=torch.int64),  # 类别从1开始
            "image_id": torch.tensor([idx])
        }
        return img_t, target


def collate_fn(batch):
    """torchvision detection 需要的 collate_fn"""
    return tuple(zip(*batch))


class DetectionModel(nn.Module):
    """封装 torchvision Faster R-CNN 或 RetinaNet，统一接口"""
    def __init__(self, num_classes=1, backbone="resnet50", pretrained=True, device="cpu", model_type="fasterrcnn"):
        super().__init__()
        self.num_classes = num_classes + 1  # +1 for background
        self.device = torch.device(device)
        self.model_type = model_type
        self.backbone_name = backbone
        
        if model_type == "fasterrcnn":
            # Faster R-CNN ResNet50 FPN v2 - 精度更高
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            # 替换分类头
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        else:
            # RetinaNet - 速度更快，单阶段
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
            self.model = retinanet_resnet50_fpn_v2(weights=weights)
            # 替换分类头
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head = RetinaNetHead(
                self.model.backbone.out_channels,
                num_anchors,
                self.num_classes
            )
        
        self.model.to(self.device)
    
    def forward(self, images, targets=None):
        """训练时传入 targets，推理时只传 images"""
        images = [img.to(self.device) for img in images]
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            return self.model(images, targets)  # 返回 loss dict
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(images)  # 返回预测列表
    
    def save_model(self, path, optimizer=None, epoch=None, loss=None):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': epoch, 'loss': loss,
            'num_classes': self.num_classes - 1, 'backbone': self.backbone_name,
            'model_type': self.model_type,
        }, path)
    
    @classmethod
    def load_model(cls, path, device="cpu", optimizer=None):
        ckpt = torch.load(path, map_location=device)
        m = cls(ckpt.get("num_classes",1), ckpt.get("backbone","resnet50"), False, device, ckpt.get("model_type","fasterrcnn"))
        m.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return m, ckpt.get("epoch",0), ckpt.get("loss")
    
    def predict(self, image_path, conf_thresh=0.3, iou_thresh=0.45, img_size=(640,640)):
        """推理接口，兼容原 predict 函数签名"""
        self.model.eval()
        img = cv2.imread(image_path)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        canvas, scale = _letterbox(img, img_size)
        img_t = torch.from_numpy(canvas).permute(2,0,1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model([img_t])[0]
        
        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        
        keep = scores > conf_thresh
        if not keep.any():
            return []
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # NMS (torchvision 已内置 NMS，但可再次过滤)
        from torchvision.ops import nms
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores)
        keep_nms = nms(boxes_t, scores_t, 0.45)
        boxes = boxes[keep_nms.numpy()]
        scores = scores[keep_nms.numpy()]
        labels = labels[keep_nms.numpy()]
        
        dets = []
        for box, score, label in zip(boxes, scores, labels):
            # 从 letterbox 坐标还原到原图
            x1, y1, x2, y2 = box
            bx1 = max(0, min(x1 / scale, ow))
            by1 = max(0, min(y1 / scale, oh))
            bx2 = max(0, min(x2 / scale, ow))
            by2 = max(0, min(y2 / scale, oh))
            dets.append({
                'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                'score': float(score),
                'class_id': int(label) - 1  # 去掉 background
            })
        return dets


def predict(model, image_path, device="cpu", conf_thresh=0.3, iou_thresh=0.45, img_size=(640,640)):
    """兼容原接口的预测函数"""
    if hasattr(model, "predict"):
        return model.predict(image_path, conf_thresh, iou_thresh, img_size)
    # 兼容旧模型
    return []


def detection_loss(cls_logits, reg_pred, targets, device):
    """兼容旧接口，实际不使用（torchvision 内部计算 loss）"""
    return torch.tensor(0.0, device=device)


# 兼容性常量
ANCHORS = [(0.375, 1.5), (0.75, 0.75), (1.5, 0.375), (0.75, 3.0), (1.5, 1.5), (3.0, 0.75), (1.5, 6.0), (3.0, 3.0), (6.0, 1.5)]
ANCHORS_T = torch.tensor(ANCHORS, dtype=torch.float32)
