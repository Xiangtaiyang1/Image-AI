# image-ai package
"""
Image AI - 图片识别标注训练推理一体化工具

包含模块:
- data_manager: 数据管理 (images-unmarked <-> images-marked JSON)
- annotation_gui: Tkinter 图形化框选标注工具
- model: PyTorch 目标检测模型 (支持多次训练/调用)
- train: 训练入口
- predict: 推理入口
"""

__version__ = "1.0.0"
__author__ = "Image AI Team"

from .data_manager import DataManager
from .model import DetectionModel, DetectionDataset, predict

__all__ = [
    "DataManager",
    "DetectionModel",
    "DetectionDataset",
    "predict",
]
