# 数据管理模块 - 管理 images-unmarked 图片与 images-marked JSON 标注
import os
import json
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class DataManager:
    """管理未标注图片与已标注 JSON 数据的交互"""
    
    def __init__(self, 
                 unmarked_dir: str = "images-unmarked",
                 marked_dir: str = "images-marked",
                 supported_ext: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.unmarked_dir = Path(unmarked_dir)
        self.marked_dir = Path(marked_dir)
        self.supported_ext = supported_ext
        
        # 确保目录存在
        self.unmarked_dir.mkdir(parents=True, exist_ok=True)
        self.marked_dir.mkdir(parents=True, exist_ok=True)
    
    def get_unmarked_images(self) -> List[Path]:
        """获取所有未标注的图片路径"""
        images = []
        for ext in self.supported_ext:
            images.extend(self.unmarked_dir.glob(f"*{ext}"))
            images.extend(self.unmarked_dir.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def get_marked_images(self) -> List[Path]:
        """获取已标注的图片路径（存在对应 JSON）"""
        marked = []
        for json_file in self.marked_dir.glob("*.json"):
            img_stem = json_file.stem
            # 尝试找到对应的图片
            for ext in self.supported_ext:
                img_path = self.unmarked_dir / f"{img_stem}{ext}"
                if img_path.exists():
                    marked.append(img_path)
                    break
        return sorted(marked)
    
    def get_unmarked_stems(self) -> List[str]:
        """获取未标注图片的文件名（不含扩展名）"""
        return [p.stem for p in self.get_unmarked_images()]
    
    def get_marked_stems(self) -> List[str]:
        """获取已标注图片的文件名"""
        return [p.stem for p in self.get_marked_images()]
    
    def get_pending_images(self) -> List[Path]:
        """获取待标注的图片（存在于 unmarked 但无对应 JSON）"""
        unmarked_stems = set(self.get_unmarked_stems())
        marked_stems = set(self.get_marked_stems())
        pending_stems = unmarked_stems - marked_stems
        
        pending = []
        for stem in sorted(pending_stems):
            for ext in self.supported_ext:
                p = self.unmarked_dir / f"{stem}{ext}"
                if p.exists():
                    pending.append(p)
                    break
        return pending
    
    def load_annotation(self, image_stem: str) -> Optional[Dict]:
        """加载指定图片的标注 JSON"""
        json_path = self.marked_dir / f"{image_stem}.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载标注失败 {json_path}: {e}")
            return None
    
    def save_annotation(self, image_stem: str, annotation: Dict) -> bool:
        """保存标注到 JSON 文件"""
        json_path = self.marked_dir / f"{image_stem}.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存标注失败 {json_path}: {e}")
            return False
    
    def delete_image(self, image_path) -> bool:
        """删除图片文件及其对应的标注 JSON。"""
        path = Path(image_path)
        try:
            if path.exists():
                path.unlink()
            json_path = self.marked_dir / f"{path.stem}.json"
            if json_path.exists():
                json_path.unlink()
            return True
        except Exception as e:
            print(f"删除图片失败 {image_path}: {e}")
            return False

    def delete_annotation(self, image_stem: str) -> bool:
        """删除标注文件"""
        json_path = self.marked_dir / f"{image_stem}.json"
        if json_path.exists():
            try:
                json_path.unlink()
                return True
            except Exception as e:
                print(f"删除标注失败 {json_path}: {e}")
                return False
        return False
    
    def create_empty_annotation(self, image_stem: str, image_path: Path) -> Dict:
        """创建空的标注模板"""
        img = cv2.imread(str(image_path))
        if img is None:
            h, w = 480, 640
        else:
            h, w = img.shape[:2]
        return {
            "image_name": image_path.name,
            "image_width": w,
            "image_height": h,
            "annotations": []
        }
    
    def get_progress(self) -> Dict:
        """获取标注进度统计"""
        total = len(self.get_unmarked_images())
        marked = len(self.get_marked_images())
        pending = total - marked
        return {
            "total": total,
            "marked": marked,
            "pending": pending,
            "progress": f"{marked}/{total}" if total > 0 else "0/0"
        }


# 标注数据格式定义
# {
#     "image_name": "test.jpg",
#     "image_width": 640,
#     "image_height": 480,
#     "annotations": [
#         {
#             "class_id": 0,
#             "class_name": "object",
#             "bbox": [x, y, width, height],  # 相对坐标 0-1
#             "bbox_abs": [x1, y1, x2, y2],   # 绝对像素坐标
#         },
#         ...
#     ]
# }

def normalize_bbox(x1: float, y1: float, x2: float, y2: float, 
                   img_w: int, img_h: int) -> List[float]:
    """将绝对坐标转为相对坐标 (x_center, y_center, width, height) 0-1"""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = abs(x2 - x1) / img_w
    bh = abs(y2 - y1) / img_h
    return [cx, cy, bw, bh]

def denormalize_bbox(cx: float, cy: float, bw: float, bh: float,
                     img_w: int, img_h: int) -> List[int]:
    """将相对坐标转为绝对坐标 [x1, y1, x2, y2]"""
    x1 = int((cx - bw/2) * img_w)
    y1 = int((cy - bh/2) * img_h)
    x2 = int((cx + bw/2) * img_w)
    y2 = int((cy + bh/2) * img_h)
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    # 简单测试
    dm = DataManager()
    print("进度:", dm.get_progress())
    print("待标注:", [str(p) for p in dm.get_pending_images()])
