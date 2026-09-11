# 图形化标注工具 - 基于 Tkinter 的图片框选标注界面
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_manager import DataManager, normalize_bbox, denormalize_bbox


class AnnotationTool:
    """Tkinter 图形化标注工具：支持鼠标框选、类别管理、JSON 保存"""
    
    def __init__(self, root: tk.Tk, unmarked_dir: str = "images-unmarked-2",
                 marked_dir: str = "images-marked-2"):
        self.root = root
        self.root.title("Image Annotation Tool - 图片标注工具")
        self.root.geometry("1400x900")
        
        self.dm = DataManager(unmarked_dir, marked_dir)
        self.classes = ["object"]  # 默认类别
        self.current_class = 0
        
        # 状态变量
        self.current_image_path = None
        self.current_image = None
        self.tk_image = None
        self.scale = 1.0
        self.display_w = 0
        self.display_h = 0
        
        # 框选状态
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.rectangles = []  # 存储当前图片的所有框
        
        self._build_ui()
        self.load_next_pending()
    
    def _build_ui(self):
        """构建 UI 界面"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="上一张 (A)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="下一张 (D)", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存标注 (S)", command=self.save_annotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除选中框 (Del)", command=self.delete_selected_rect).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="清空当前图标注", command=self.clear_all_rects).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除当前图片", command=self.delete_current_image).pack(side=tk.LEFT, padx=2)
        
        # 进度显示
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(toolbar, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=20)
        self.update_progress()
        
        # 主画布区域
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 鼠标事件绑定
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # 右侧面板
        right_panel = ttk.Frame(self.root, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # 类别管理
        class_frame = ttk.LabelFrame(right_panel, text="类别管理")
        class_frame.pack(fill=tk.X, pady=5)
        
        self.class_listbox = tk.Listbox(class_frame, height=6)
        self.class_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.class_listbox.bind("<<ListboxSelect>>", self.on_class_select)
        self.refresh_class_list()
        
        class_btn_frame = ttk.Frame(class_frame)
        class_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(class_btn_frame, text="添加类别", command=self.add_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(class_btn_frame, text="删除类别", command=self.delete_class).pack(side=tk.LEFT, padx=2)
        
        # 标注列表
        ann_frame = ttk.LabelFrame(right_panel, text="当前图片标注")
        ann_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.ann_tree = ttk.Treeview(ann_frame, columns=("class", "bbox"), show="headings", height=15)
        self.ann_tree.heading("class", text="类别")
        self.ann_tree.heading("bbox", text="边界框 (x1,y1,x2,y2)")
        self.ann_tree.column("class", width=80)
        self.ann_tree.column("bbox", width=180)
        self.ann_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ann_tree.bind("<<TreeviewSelect>>", self.on_ann_select)
        
        # 缩放控制
        zoom_frame = ttk.LabelFrame(right_panel, text="缩放控制")
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Button(zoom_frame, text="放大 (+)", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="缩小 (-)", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="适应窗口", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="原始大小 (1:1)", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 键盘快捷键
        self.root.bind("<Key-a>", lambda e: self.prev_image())
        self.root.bind("<Key-d>", lambda e: self.next_image())
        self.root.bind("<Key-s>", lambda e: self.save_annotation())
        self.root.bind("<Delete>", lambda e: self.delete_selected_rect())
        self.root.bind("<Control-Delete>", lambda e: self.delete_current_image())
        self.root.bind("<Control-s>", lambda e: self.save_annotation())
        self.root.bind("<Key-1>", lambda e: self.set_class(0))
        self.root.bind("<Key-2>", lambda e: self.set_class(1))
        self.root.bind("<Key-3>", lambda e: self.set_class(2))
        self.root.bind("<Key-4>", lambda e: self.set_class(3))
        self.root.bind("<Key-5>", lambda e: self.set_class(4))
        self.root.bind("<Key-6>", lambda e: self.set_class(5))
        self.root.bind("<Key-7>", lambda e: self.set_class(6))
        self.root.bind("<Key-8>", lambda e: self.set_class(7))
        self.root.bind("<Key-9>", lambda e: self.set_class(8))
        self.root.bind("<Key-0>", lambda e: self.set_class(9))
        self.root.bind("<Key-plus>", lambda e: self.zoom(1.2))
        self.root.bind("<Key-minus>", lambda e: self.zoom(0.8))
        self.root.bind("<Key-0>", lambda e: self.reset_zoom())
        
        # 右键菜单
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="删除该框", command=self.delete_selected_rect)
        self.context_menu.add_command(label="编辑类别", command=self.edit_rect_class)
        self.canvas.bind("<Button-3>", self.show_context_menu)
    
    def refresh_class_list(self):
        """刷新类别列表"""
        self.class_listbox.delete(0, tk.END)
        for i, cls in enumerate(self.classes):
            self.class_listbox.insert(tk.END, f"{i}: {cls}")
        if self.current_class < len(self.classes):
            self.class_listbox.selection_set(self.current_class)
    
    def on_class_select(self, event):
        """选择类别"""
        sel = self.class_listbox.curselection()
        if sel:
            self.current_class = sel[0]
            self.update_status(f"当前类别: {self.classes[self.current_class]}")
    
    def add_class(self):
        """添加新类别"""
        name = tk.simpledialog.askstring("添加类别", "请输入类别名称:")
        if name and name.strip():
            self.classes.append(name.strip())
            self.refresh_class_list()
            self.current_class = len(self.classes) - 1
            self.class_listbox.selection_set(self.current_class)
    
    def delete_class(self):
        """删除类别"""
        if len(self.classes) <= 1:
            messagebox.showwarning("警告", "至少保留一个类别")
            return
        sel = self.class_listbox.curselection()
        if sel:
            idx = sel[0]
            del self.classes[idx]
            self.refresh_class_list()
            if self.current_class >= len(self.classes):
                self.current_class = len(self.classes) - 1
            self.class_listbox.selection_set(self.current_class)
    
    def set_class(self, idx):
        """快捷键设置类别"""
        if 0 <= idx < len(self.classes):
            self.current_class = idx
            self.class_listbox.selection_clear(0, tk.END)
            self.class_listbox.selection_set(idx)
            self.update_status(f"当前类别: {self.classes[idx]} (快捷键 {idx})")
    
    def update_progress(self):
        """更新进度显示"""
        prog = self.dm.get_progress()
        self.progress_var.set(f"进度: {prog['progress']}  待标注: {prog['pending']}")
    
    def update_status(self, msg: str):
        """更新状态栏"""
        self.status_var.set(msg)
        self.root.update_idletasks()
    
    def load_next_pending(self):
        """加载下一张待标注图片"""
        pending = self.dm.get_pending_images()
        if not pending:
            # 如果没有待标注，加载第一张
            all_images = self.dm.get_unmarked_images()
            if all_images:
                self.current_image_path = all_images[0]
            else:
                messagebox.showinfo("完成", "所有图片已标注完成！")
                return
        else:
            self.current_image_path = pending[0]
        self.load_image()
    
    def load_image(self):
        """加载并显示图片"""
        if not self.current_image_path or not self.current_image_path.exists():
            return
        
        # 加载图片
        self.current_image = cv2.imread(str(self.current_image_path))
        if self.current_image is None:
            self.update_status(f"无法加载图片: {self.current_image_path}")
            return
        
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w = self.current_image.shape[:2]
        
        # 加载已有标注
        stem = self.current_image_path.stem
        ann = self.dm.load_annotation(stem)
        self.rectangles = []
        if ann and "annotations" in ann:
            for a in ann["annotations"]:
                # 转换为绝对坐标
                if "bbox_abs" in a:
                    x1, y1, x2, y2 = a["bbox_abs"]
                elif "bbox" in a:
                    x1, y1, x2, y2 = denormalize_bbox(
                        a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3], w, h
                    )
                else:
                    continue
                self.rectangles.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class_id": a.get("class_id", 0),
                    "class_name": a.get("class_name", self.classes[0] if self.classes else "object")
                })
        
        self.fit_to_window()
        self.update_annotation_list()
        self.update_progress()
        self.update_status(f"已加载: {self.current_image_path.name} ({w}x{h})  标注框: {len(self.rectangles)} 个")
    
    def fit_to_window(self):
        """适应窗口大小"""
        if self.current_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            self.root.after(100, self.fit_to_window)
            return
        
        img_h, img_w = self.current_image.shape[:2]
        scale_w = (canvas_w - 20) / img_w
        scale_h = (canvas_h - 20) / img_h
        self.scale = min(scale_w, scale_h, 1.0)
        self._redraw()
    
    def reset_zoom(self):
        """重置缩放为 1:1"""
        self.scale = 1.0
        self._redraw()
    
    def zoom(self, factor: float):
        """缩放"""
        self.scale *= factor
        self.scale = max(0.1, min(self.scale, 5.0))
        self._redraw()
    
    def _redraw(self):
        """重绘画布"""
        if self.current_image is None:
            return
        
        img = self.current_image.copy()
        h, w = img.shape[:2]
        
        # 计算显示尺寸
        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)
        display_img = cv2.resize(img, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        
        # 绘制已有矩形框
        for i, rect in enumerate(self.rectangles):
            x1 = int(rect["x1"] * self.scale)
            y1 = int(rect["y1"] * self.scale)
            x2 = int(rect["x2"] * self.scale)
            y2 = int(rect["y2"] * self.scale)
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, thickness)
            # 绘制类别标签
            label = f"{rect['class_name']}"
            cv2.putText(display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6 * self.scale, color, max(1, int(2 * self.scale)))
        
        # 绘制正在绘制的框
        if self.drawing and self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 转为 Tkinter 可用格式
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        
        # 清空画布并绘制
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, self.display_w, self.display_h))
    
    # 鼠标事件处理
    def on_mouse_down(self, event):
        """鼠标按下开始画框"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = [event.x, event.y, event.x, event.y]
        self._redraw()
    
    def on_mouse_drag(self, event):
        """鼠标拖动调整框"""
        if self.drawing:
            self.current_rect[2] = event.x
            self.current_rect[3] = event.y
            self._redraw()
    
    def on_mouse_up(self, event):
        """鼠标松开完成画框"""
        if self.drawing:
            self.drawing = False
            x1, y1, x2, y2 = self.current_rect
            # 确保 x1 < x2, y1 < y2
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            # 过滤太小的框
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                # 转换为原图坐标
                rx1 = int(x1 / self.scale)
                ry1 = int(y1 / self.scale)
                rx2 = int(x2 / self.scale)
                ry2 = int(y2 / self.scale)
                # 限制在图片范围内
                img_h, img_w = self.current_image.shape[:2]
                rx1 = max(0, min(rx1, img_w - 1))
                rx2 = max(0, min(rx2, img_w - 1))
                ry1 = max(0, min(ry1, img_h - 1))
                ry2 = max(0, min(ry2, img_h - 1))
                
                self.rectangles.append({
                    "x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2,
                    "class_id": self.current_class,
                    "class_name": self.classes[self.current_class] if self.classes else "object"
                })
                self.update_annotation_list()
                self.update_status(f"添加标注框: {self.classes[self.current_class]} ({rx1},{ry1},{rx2},{ry2})")
            self.current_rect = None
            self._redraw()
    
    def on_mouse_move(self, event):
        """鼠标移动显示坐标"""
        if self.current_image is not None:
            x = int(event.x / self.scale)
            y = int(event.y / self.scale)
            self.update_status(f"坐标: ({x}, {y})  缩放: {self.scale:.2f}x")
    
    def update_annotation_list(self):
        """更新标注列表"""
        for item in self.ann_tree.get_children():
            self.ann_tree.delete(item)
        for i, rect in enumerate(self.rectangles):
            self.ann_tree.insert("", tk.END, iid=str(i), values=(
                rect["class_name"],
                f"({rect['x1']},{rect['y1']},{rect['x2']},{rect['y2']})"
            ))
    
    def on_ann_select(self, event):
        """选择标注列表项"""
        sel = self.ann_tree.selection()
        if sel:
            idx = int(sel[0])
            # 高亮显示对应框
            self._redraw()
    
    def delete_current_image(self):
        """删除当前图片文件及其标注，并跳到下一张待标注图片。"""
        if not self.current_image_path:
            return
        name = self.current_image_path.name
        if self.dm.delete_image(self.current_image_path):
            self.update_status("已删除图片: " + name)
            self.current_image_path = None
            self.canvas.delete("all")
            self.rectangles = []
            self.update_annotation_list()
            self.update_progress()
            self.load_next_pending()
        else:
            messagebox.showerror("错误", "删除图片失败")
    def delete_selected_rect(self):
        """删除选中的标注框"""
        sel = self.ann_tree.selection()
        if sel:
            idx = int(sel[0])
            del self.rectangles[idx]
            self.update_annotation_list()
            self._redraw()
            self.update_status(f"已删除标注框 {idx}")
    
    def clear_all_rects(self):
        """清空所有标注框"""
        if self.rectangles:
            self.rectangles.clear()
            self.update_annotation_list()
            self._redraw()
            self.update_status("已清空所有标注框")
    
    def edit_rect_class(self):
        """编辑选中框的类别"""
        sel = self.ann_tree.selection()
        if sel:
            idx = int(sel[0])
            new_class = tk.simpledialog.askstring("编辑类别", "请输入新类别名称:", 
                                                  initialvalue=self.rectangles[idx]["class_name"])
            if new_class and new_class.strip():
                self.rectangles[idx]["class_name"] = new_class.strip()
                self.rectangles[idx]["class_id"] = self.classes.index(new_class.strip()) if new_class.strip() in self.classes else 0
                self.update_annotation_list()
                self._redraw()
    
    def show_context_menu(self, event):
        """显示右键菜单"""
        self.context_menu.tk_popup(event.x_root, event.y_root)
    
    def save_annotation(self):
        """保存当前图片标注"""
        if not self.current_image_path:
            return
        
        stem = self.current_image_path.stem
        h, w = self.current_image.shape[:2]
        
        annotations = []
        for rect in self.rectangles:
            # 同时保存绝对坐标和相对坐标
            cx, cy, bw, bh = normalize_bbox(rect["x1"], rect["y1"], rect["x2"], rect["y2"], w, h)
            annotations.append({
                "class_id": rect["class_id"],
                "class_name": rect["class_name"],
                "bbox": [cx, cy, bw, bh],  # YOLO 格式相对坐标
                "bbox_abs": [rect["x1"], rect["y1"], rect["x2"], rect["y2"]]  # 绝对坐标
            })
        
        annotation = {
            "image_name": self.current_image_path.name,
            "image_width": w,
            "image_height": h,
            "annotations": annotations
        }
        
        if self.dm.save_annotation(stem, annotation):
            self.update_progress()
            self.update_status(f"标注已保存: {stem}.json ({len(annotations)} 个框)")
        else:
            messagebox.showerror("错误", "保存标注失败")
    
    def prev_image(self):
        """上一张图片"""
        if not self.current_image_path:
            self.load_next_pending()
            return
        
        all_images = self.dm.get_unmarked_images()
        if not all_images:
            return
        
        current_idx = -1
        for i, p in enumerate(all_images):
            if p.stem == self.current_image_path.stem:
                current_idx = i
                break
        
        if current_idx > 0:
            self.current_image_path = all_images[current_idx - 1]
            self.load_image()
        else:
            self.update_status("已经是第一张")
    
    def next_image(self):
        """下一张图片：跳到下一张还未标注的图片。"""
        pending = self.dm.get_pending_images()
        if not pending:
            messagebox.showinfo("完成", "所有图片都已标注完成！")
            self.update_status("所有图片已标注完成")
            return

        # 所有未标注图片的 stem 集合，当前图片之后优先
        pending_stems = [p.stem for p in pending]

        if self.current_image_path and self.current_image_path.stem in pending_stems:
            cur = self.current_image_path.stem
            idx = pending_stems.index(cur)
            if idx < len(pending) - 1:
                self.current_image_path = pending[idx + 1]
            else:
                self.current_image_path = pending[0]  # 已到末尾则回绕到第一张待标注
                self.update_status("已是最后一张待标注，回绕到开头")
        else:
            # 当前图片已标注（不在待标注列表），直接拿第一张待标注
            self.current_image_path = pending[0]

        self.load_image()
    
    def run(self):
        """启动主循环"""
        self.root.mainloop()


def main():
    """主入口"""
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
