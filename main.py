import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _parse_kv(args):
    result = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:].replace("-", "_")
            if key == "img_size" and i + 2 < len(args) and not args[i+1].startswith("--") and not args[i+2].startswith("--"):
                result[key] = f"{args[i+1]} {args[i+2]}"
                i += 3
            elif i + 1 < len(args) and not args[i+1].startswith("--"):
                result[key] = args[i+1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            i += 1
    return result

def _to_bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("1", "true", "yes", "on")

def train(marked_dir="images-marked", unmarked_dir="images-unmarked", model_dir="saved_models",
      epochs=50, batch_size=4, lr=1e-4, img_size=(640,640), backbone="resnet50",
      pretrained=True, resume=None, device="auto", num_workers=0, save_interval=5, no_plot=False, model_type="fasterrcnn"):
    os.environ["MARKED_DIR"] = marked_dir
    os.environ["UNMARKED_DIR"] = unmarked_dir
    os.environ["MODEL_DIR"] = model_dir
    os.environ["EPOCHS"] = str(epochs)
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["LR"] = str(lr)
    os.environ["IMG_SIZE"] = f"{img_size[0]} {img_size[1]}"
    os.environ["BACKBONE"] = backbone
    os.environ["PRETRAINED"] = "1" if pretrained else "0"
    os.environ["RESUME"] = resume or ""
    os.environ["DEVICE"] = device
    os.environ["NUM_WORKERS"] = str(num_workers)
    os.environ["SAVE_INTERVAL"] = str(save_interval)
    os.environ["NO_PLOT"] = "1" if no_plot else "0"
    os.environ["MODEL_TYPE"] = model_type
    import train
    train.main()

def predict(model_path, input_path, output_dir="output", conf=0.5, iou=0.5,
          img_size=(640,640), device="auto", save_vis=True, save_json=True, classes=None):
    os.environ["MODEL_PATH"] = model_path
    os.environ["INPUT_PATH"] = input_path
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["CONF"] = str(conf)
    os.environ["IOU"] = str(iou)
    os.environ["PRED_IMG_SIZE"] = f"{img_size[0]} {img_size[1]}"
    os.environ["PRED_DEVICE"] = device
    os.environ["SAVE_VIS"] = "1" if save_vis else "0"
    os.environ["SAVE_JSON"] = "1" if save_json else "0"
    os.environ["CLASSES"] = classes or ""
    import predict
    predict.main()

def annotate(unmarked_dir="images-unmarked", marked_dir="images-marked"):
    os.environ["ANN_UNMARKED_DIR"] = unmarked_dir
    os.environ["ANN_MARKED_DIR"] = marked_dir
    import annotation_gui
    annotation_gui.main()

def _parse_kv(args):
    result = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:].replace("-", "_")
            if key == "img_size" and i + 2 < len(args) and not args[i+1].startswith("--") and not args[i+2].startswith("--"):
                result[key] = f"{args[i+1]} {args[i+2]}"
                i += 3
            elif i + 1 < len(args) and not args[i+1].startswith("--"):
                result[key] = args[i+1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            i += 1
    return result

def _to_bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("1", "true", "yes", "on")

def _cmd_train(argv):
    kwargs = _parse_kv(argv)
    for k, v in kwargs.items():
        if k in ("epochs", "batch_size", "num_workers", "save_interval"):
            kwargs[k] = int(v)
        elif k in ("lr",):
            kwargs[k] = float(v)
        elif k in ("pretrained", "no_plot"):
            kwargs[k] = _to_bool(v)
        elif k == "img_size":
            # img_size 在 _parse_kv 中已处理为 "W H" 字符串
            pass
    train(**kwargs)

def _cmd_predict(argv):
    if len(argv) < 2:
        print("predict 用法: python main.py predict <model> <input> [output] [--conf 0.5] ...")
        return
    model_path = argv[0]
    input_path = argv[1]
    output_dir = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else "output"
    remaining = argv[3:] if len(argv) > 2 and not argv[2].startswith("--") else argv[2:]
    kwargs = _parse_kv(remaining)
    for k, v in kwargs.items():
        if k in ("conf", "iou"):
            kwargs[k] = float(v)
    predict(model_path, input_path, output_dir, **kwargs)

def _cmd_annotate(argv):
    kwargs = _parse_kv(argv)
    annotate(**kwargs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py train|predict|annotate [参数]")
        print("或: import main; main.train() / main.predict() / main.annotate()")
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "train":
        _cmd_train(sys.argv[2:])
    elif cmd == "predict":
        _cmd_predict(sys.argv[2:])
    elif cmd == "annotate":
        _cmd_annotate(sys.argv[2:])
    else:
        print(f"未知命令: {cmd}")
        sys.exit(1)
