import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DetectionModel, DetectionDataset, collate_fn

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

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

def main():
    marked_dir = get_env("MARKED_DIR", "images-marked")
    unmarked_dir = get_env("UNMARKED_DIR", "images-unmarked")
    model_dir = get_env("MODEL_DIR", "saved_models")
    epochs = get_env_int("EPOCHS", 50)
    batch_size = get_env_int("BATCH_SIZE", 4)
    lr = get_env_float("LR", 1e-4)
    img_size = get_env_tuple("IMG_SIZE", (640, 640))
    backbone = get_env("BACKBONE", "resnet50")
    pretrained = get_env_bool("PRETRAINED", True)
    resume = get_env("RESUME", None)
    device = get_env("DEVICE", "auto")
    num_workers = get_env_int("NUM_WORKERS", 0)
    save_interval = get_env_int("SAVE_INTERVAL", 5)
    no_plot = get_env_bool("NO_PLOT", False)
    model_type = get_env("MODEL_TYPE", "fasterrcnn")
    
    # 强制使用本地缓存，避免权限问题
    os.environ.setdefault("TORCH_HOME", os.path.join(os.getcwd(), "torch_cache"))
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
    
    device = "cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)
    print(f"使用设备: {device}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    dataset = DetectionDataset(marked_dir, unmarked_dir, img_size)
    if len(dataset) == 0:
        print("错误: 没有找到标注数据")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=num_workers)
    
    import json
    from pathlib import Path
    json_files = list(Path(marked_dir).glob("*.json"))
    num_classes = 1
    if json_files:
        with open(json_files[0], encoding="utf-8") as f:
            ann = json.load(f)
            class_ids = {a.get("class_id",0) for a in ann.get("annotations",[])}
            num_classes = max(class_ids)+1 if class_ids else 1
    print(f"类别数: {num_classes}")
    
    model = DetectionModel(num_classes, backbone, True, device, model_type)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    if resume and os.path.exists(resume):
        print(f"加载检查点: {resume}")
        model, start_epoch, _ = DetectionModel.load_model(resume, device, optimizer)
        print(f"从 epoch {start_epoch} 继续训练 (optimizer状态已恢复)")
    
    loss_history = []
    fig = ax = None
    if HAS_MPL and not no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Training Loss (Real-time)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        plt.show(block=False)
        plt.pause(0.5)
    
    print(f"开始训练 {epochs} epochs (model_type={model_type})")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for images, targets in dataloader:
            images = list(images)
            targets = list(targets)
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if HAS_MPL and not no_plot and fig is not None:
                fig.canvas.flush_events()
                plt.pause(0.001)
        
        avg_loss = total_loss / max(1, num_batches)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        if HAS_MPL and not no_plot and ax is not None:
            ax.clear()
            epochs_list = list(range(1, len(loss_history)+1))
            ax.plot(epochs_list, loss_history, 'b-', linewidth=2, label='Train Loss')
            ax.set_title(f'Training Loss (Real-time) - Epoch {epoch+1}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)
        
        scheduler.step()
        
        if (epoch+1) % save_interval == 0:
            os.makedirs(model_dir, exist_ok=True)
            model.save_model(os.path.join(model_dir, f"checkpoint_{epoch+1}.pth"),
                           optimizer, epoch+1, total_loss/max(1,num_batches))
    
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "final_model.pth"),
                       optimizer, epochs, total_loss/max(1,num_batches))
    print("训练完成")
    if HAS_MPL and not no_plot and fig is not None:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
