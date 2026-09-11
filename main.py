# 主程序入口
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Image AI - 图片识别标注训练推理一体化工具")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")
    
    ann = subparsers.add_parser("annotate", help="图形化标注工具")
    ann.add_argument("--unmarked-dir", default="images-unmarked")
    ann.add_argument("--marked-dir", default="images-marked")
    
    train = subparsers.add_parser("train", help="训练模型")
    train.add_argument("--marked-dir", default="images-marked")
    train.add_argument("--unmarked-dir", default="images-unmarked")
    train.add_argument("--model-dir", default="saved_models")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--img-size", type=int, nargs=2, default=[640, 640])
    train.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--resume", type=str, default=None)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--save-interval", type=int, default=5)
    train.add_argument("--no-plot", action="store_true", help="禁用实时损失曲线")
    
    pred = subparsers.add_parser("predict", help="推理预测")
    pred.add_argument("--model", required=True)
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", default="output")
    pred.add_argument("--conf", type=float, default=0.5)
    pred.add_argument("--iou", type=float, default=0.5)
    pred.add_argument("--img-size", type=int, nargs=2, default=[640, 640])
    pred.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pred.add_argument("--save-vis", action="store_true")
    pred.add_argument("--save-json", action="store_true")
    pred.add_argument("--classes", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == "annotate":
        from annotation_gui import main as gui_main
        sys.argv = [sys.argv[0], "--unmarked-dir", args.unmarked_dir, "--marked-dir", args.marked_dir]
        gui_main()
    elif args.mode == "train":
        from train import main as train_main
    elif args.mode == "predict":
        from predict import main as predict_main
    else:
        parser.print_help()
        print("示例用法:")
        print("  python main.py annotate --unmarked-dir images-unmarked --marked-dir images-marked")
        print("  python main.py train --marked-dir images-marked --unmarked-dir images-unmarked --epochs 50")
        print("  python main.py predict --model saved_models/best_model.pth --input test.jpg --output output")
        return
    
    # 将参数转发给子命令
    new_argv = [sys.argv[0]]
    for k, v in vars(args).items():
        if k == "mode" or v is None:
            continue
        if isinstance(v, bool):
            if v:
                new_argv.append(f"--{k.replace('_', '-')}")
        elif isinstance(v, list):
            new_argv.append(f"--{k.replace('_', '-')}")
            new_argv.extend(str(x) for x in v)
        else:
            new_argv.append(f"--{k.replace('_', '-')}")
            new_argv.append(str(v))
    sys.argv = new_argv
    
    if args.mode == "train":
        from train import main as train_main
        train_main()
    elif args.mode == "predict":
        from predict import main as predict_main
        predict_main()

if __name__ == "__main__":
    main()