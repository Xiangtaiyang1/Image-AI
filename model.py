import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.models as m_models
import cv2, numpy as np, json, math, os
from pathlib import Path
from typing import List, Dict

ANCHORS = [
    (0.375, 1.5), (0.75, 0.75), (1.5, 0.375),
    (0.75, 3.0), (1.5, 1.5), (3.0, 0.75),
    (1.5, 6.0), (3.0, 3.0), (6.0, 1.5),
]
ANCHORS_T = torch.tensor(ANCHORS, dtype=torch.float32)
STRIDE = 32

class DetectionModel(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet18', pretrained=False, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.backbone_name = backbone
        try:
            bm = getattr(m_models, backbone)(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(bm.children())[:-2])
            self.backbone_out = 512
        except Exception:
            self.backbone = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            )
            self.backbone_out = 256
        self.proj = nn.Conv2d(self.backbone_out, 512, 1)
        self.fpn = nn.Sequential(nn.Conv2d(512,256,1), nn.ReLU(), nn.Conv2d(256,256,3,padding=1), nn.ReLU())
        self.cls_head = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.Conv2d(256,9,3,padding=1))
        self.reg_head = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.Conv2d(256,36,3,padding=1))
        self.to(self.device)

    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        f = self.fpn(f)
        cls = self.cls_head(f).view(f.shape[0], -1, 1)
        reg = self.reg_head(f).view(f.shape[0], -1, 4)
        return cls, reg

    def save_model(self, path, optimizer=None, epoch=None, loss=None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': epoch, 'loss': loss,
            'num_classes': self.num_classes, 'backbone': self.backbone_name,
        }, path)

    @classmethod
    def load_model(cls, path, device='cpu', optimizer=None):
        ckpt = torch.load(path, map_location=device)
        m = cls(ckpt.get('num_classes',1), ckpt.get('backbone','resnet18'), False, device)
        m.load_state_dict(ckpt['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return m, ckpt.get('epoch',0), ckpt.get('loss')

def _letterbox(image, img_size):
    w, h = img_size
    ih, iw = image.shape[:2]
    scale = min(h/ih, w/iw)
    nw, nh = max(1,int(iw*scale)), max(1,int(ih*scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def _build_anchor_grid(W, H):
    Gx = math.ceil(W/STRIDE); Gy = math.ceil(H/STRIDE)
    gys = torch.arange(Gy, dtype=torch.float32)
    gxs = torch.arange(Gx, dtype=torch.float32)
    gy, gx = torch.meshgrid(gys, gxs, indexing="ij")
    gy = gy.reshape(-1, 1); gx = gx.reshape(-1, 1)
    aw = ANCHORS_T[:, 0].reshape(1, -1)
    ah = ANCHORS_T[:, 1].reshape(1, -1)
    cx = (gx + 0.5) * STRIDE / W
    cx = cx.expand(-1, len(ANCHORS)).reshape(-1)
    cy = (gy + 0.5) * STRIDE / H
    cy = cy.expand(-1, len(ANCHORS)).reshape(-1)
    w = (aw * STRIDE / W).expand(Gx*Gy, -1).reshape(-1)
    h = (ah * STRIDE / H).expand(Gx*Gy, -1).reshape(-1)
    return cx, cy, w, h


class DetectionDataset(Dataset):
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
                cid = int(a.get('class_id', 0))
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
                boxes.append((x1,y1,x2,y2,cid))
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
        W,H = self.img_size
        img_t = torch.from_numpy(canvas).permute(2,0,1).float()/255.0
        Gx = math.ceil(W/STRIDE); Gy = math.ceil(H/STRIDE)
        N = Gy*Gx*len(ANCHORS)
        cls_t = torch.zeros(N, dtype=torch.long)
        reg_t = torch.zeros(N, 4, dtype=torch.float32)
        assigned = torch.zeros(N, dtype=torch.bool)
        for (x1,y1,x2,y2,cid) in boxes:
            x1,y1 = int(x1*scale), int(y1*scale)
            x2,y2 = int(x2*scale), int(y2*scale)
            cxp = (x1+x2)/2.0; cyp = (y1+y2)/2.0
            bw = max(1, x2-x1); bh = max(1, y2-y1)
            gx = min(int(cxp/STRIDE), Gx-1); gy = min(int(cyp/STRIDE), Gy-1)
            aw = ANCHORS_T[:, 0] * STRIDE
            ah = ANCHORS_T[:, 1] * STRIDE
            r = torch.abs(torch.log(torch.tensor(bw, dtype=torch.float32) / torch.clamp(aw, min=1))) + torch.abs(torch.log(torch.tensor(bh, dtype=torch.float32) / torch.clamp(ah, min=1)))
            best_a = int(torch.argmin(r))
            base = (gy*Gx + gx)*len(ANCHORS) + best_a
            aw_s, ah_s = ANCHORS[best_a]
            reg_t[base, 0] = (cxp - gx*STRIDE) / max(aw_s*STRIDE, 1)
            reg_t[base, 1] = (cyp - gy*STRIDE) / max(ah_s*STRIDE, 1)
            reg_t[base, 2] = math.log(max(bw,1) / max(aw_s*STRIDE, 1))
            reg_t[base, 3] = math.log(max(bh,1) / max(ah_s*STRIDE, 1))
            cls_t[base] = 1
            assigned[base] = True
        return img_t, {"cls": cls_t, "reg": reg_t, "assigned": assigned}


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    return imgs, [b[1] for b in batch]


def detection_loss(cls_logits, reg_pred, targets, device):
    cls_tgt = torch.stack([t["cls"] for t in targets]).to(cls_logits.dtype)
    reg_tgt = torch.stack([t["reg"] for t in targets])
    pos_mask = torch.stack([t["assigned"] for t in targets]).bool()

    cls_logits = cls_logits.squeeze(-1)
    pos = pos_mask.float()
    neg_mask = ~pos_mask
    keep_weight = pos.clone()
    pos_count = max(1, int(pos.sum().item()))
    neg_count = int(neg_mask.sum().item())
    if neg_count > pos_count * 3:
        neg_idx = neg_mask.nonzero(as_tuple=False)
        perm = torch.randperm(neg_idx.shape[0])
        keep_neg_idx = neg_idx[perm[:pos_count*3]]
        keep_weight[keep_neg_idx[:,0], keep_neg_idx[:,1]] = 1.0
    weight = keep_weight
    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        cls_logits, pos, weight=weight, reduction="sum"
    ) / weight.sum().clamp(min=1)

    if pos_mask.any():
        reg_loss = nn.functional.smooth_l1_loss(
            reg_pred[pos_mask], reg_tgt[pos_mask], reduction="sum"
        ) / pos_mask.sum().clamp(min=1)
    else:
        reg_loss = reg_pred.sum() * 0.0
    return cls_loss + reg_loss


def predict(model, image_path, device="cpu", conf_thresh=0.3, iou_thresh=0.45, img_size=(640,640)):
    model.eval()
    img = cv2.imread(image_path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    canvas, scale = _letterbox(img, img_size)
    W,H = img_size
    img_t = torch.from_numpy(canvas).permute(2,0,1).float()/255.0
    img_t = img_t.unsqueeze(0).to(model.device)

    with torch.no_grad():
        cls_logits, reg = model(img_t)
        cls_probs = torch.sigmoid(cls_logits.squeeze(-1))
        reg = reg.squeeze(0)

    anc_cx, anc_cy, anc_w, anc_h = _build_anchor_grid(W, H)
    anc_cx = anc_cx.to(reg.device); anc_cy = anc_cy.to(reg.device)
    anc_w = anc_w.to(reg.device); anc_h = anc_h.to(reg.device)

    scores = cls_probs.squeeze(0)
    keep = scores > conf_thresh
    if not keep.any():
        return []

    scores = scores[keep]
    reg = reg[keep]
    anc_cx = anc_cx[keep]
    anc_cy = anc_cy[keep]
    anc_w = anc_w[keep]
    anc_h = anc_h[keep]

    tx, ty, tw, th = reg[:,0], reg[:,1], reg[:,2], reg[:,3]
    cx = (anc_cx + tx * anc_w).clamp(0, 1)
    cy = (anc_cy + ty * anc_h).clamp(0, 1)
    w  = (anc_w * torch.exp(tw)).clamp(0, 1)
    h  = (anc_h * torch.exp(th)).clamp(0, 1)

    x1 = (cx - w/2).clamp(0, 1)
    y1 = (cy - h/2).clamp(0, 1)
    x2 = (cx + w/2).clamp(0, 1)
    y2 = (cy + h/2).clamp(0, 1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    keep_nms = torch.ops.torchvision.nms(boxes, scores, iou_thresh)

    dets = []
    for idx in keep_nms:
        dets.append({
            'bbox': [float(x1[idx])*W, float(y1[idx])*H, float(x2[idx])*W, float(y2[idx])*H],
            'score': float(scores[idx]),
            'class_id': 0
        })
    return dets