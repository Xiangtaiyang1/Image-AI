# 训练入口脚本
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DetectionModel, DetectionDataset, collate_fn, detection_loss

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

def main():
    parser = argparse.ArgumentParser(description="训练目标检测模型")
    parser.add_argument("--marked-dir", default="images-marked")
    parser.add_argument("--unmarked-dir", default="images-unmarked")
    parser.add_argument("--model-dir", default="saved_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true", help="禁用实时损失曲线")
    args = parser.parse_args()

    device = "cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    print(f"使用设备: {device}")

    os.makedirs(args.model_dir, exist_ok=True)

    dataset = DetectionDataset(args.marked_dir, args.unmarked_dir, tuple(args.img_size))
    if len(dataset) == 0:
        print("错误: 没有找到标注数据")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.num_workers)

    import json
    from pathlib import Path
    json_files = list(Path(args.marked_dir).glob("*.json"))
    num_classes = 1
    if json_files:
        with open(json_files[0], encoding="utf-8") as f:
            ann = json.load(f)
            class_ids = {a.get("class_id",0) for a in ann.get("annotations",[])}
            num_classes = max(class_ids)+1 if class_ids else 1
    print(f"类别数: {num_classes}")

    model = DetectionModel(num_classes, args.backbone, args.pretrained, device)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"加载检查点: {args.resume}")
        model, start_epoch, _ = DetectionModel.load_model(args.resume, device, optimizer)
        print(f"从 epoch {start_epoch} 继续训练 (optimizer状态已恢复)")

    # 实时损失曲线
    loss_history = []
    fig = ax = None
    if HAS_MPL and not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Training Loss (Real-time)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        plt.show(block=False)
        plt.pause(0.5)

    print(f"开始训练 {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            cls_logits, bbox_reg = model(images)
            loss = detection_loss(cls_logits, bbox_reg, targets, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            # 每批次刷新GUI事件，让窗口始终响应
            if HAS_MPL and not args.no_plot and fig is not None:
                fig.canvas.flush_events()
                plt.pause(0.001)

        avg_loss = total_loss / max(1, num_batches)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 更新实时损失曲线
        if HAS_MPL and not args.no_plot and ax is not None:
            ax.clear()
            epochs = list(range(1, len(loss_history)+1))
            ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Train Loss')
            ax.set_title(f'Training Loss (Real-time) - Epoch {epoch+1}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)

        scheduler.step()

        if (epoch+1) % args.save_interval == 0:
            os.makedirs(args.model_dir, exist_ok=True)
            model.save_model(os.path.join(args.model_dir, f"checkpoint_{epoch+1}.pth"),
                           optimizer, epoch+1, total_loss/max(1,num_batches))

    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, "final_model.pth"),
                       optimizer, args.epochs, total_loss/max(1,num_batches))
    print("训练完成")
    if HAS_MPL and not args.no_plot and fig is not None:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
