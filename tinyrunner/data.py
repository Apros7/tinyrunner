"""Dataset loading for RF-DETR. Supports COCO JSON and YOLO formats."""
import json, os, random
import numpy as np
from PIL import Image
from tinygrad import Tensor

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Augmentation helpers ──────────────────────────────────────────────────────

def _resize(img, boxes, size):
  """Resize PIL image + boxes (cxcywh normalised) to square size."""
  w0, h0 = img.size
  img = img.resize((size, size), Image.BILINEAR)
  return img, boxes  # boxes already normalised → no change needed

def _hflip(img, boxes):
  img = img.transpose(Image.FLIP_LEFT_RIGHT)
  boxes = boxes.copy(); boxes[:, 0] = 1.0 - boxes[:, 0]
  return img, boxes

def _to_tensor(img):
  arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
  arr = (arr - MEAN) / STD
  return arr.transpose(2, 0, 1)  # (3,H,W)

def augment(img, boxes, labels, img_size, training=True):
  if len(boxes) == 0:
    boxes = np.zeros((0, 4), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)
  if training and random.random() < 0.5:
    img, boxes = _hflip(img, boxes)
  img, boxes = _resize(img, boxes, img_size)
  return _to_tensor(img), boxes.astype(np.float32), labels.astype(np.int64)

# ── COCO Dataset ──────────────────────────────────────────────────────────────

class COCODataset:
  """Loads COCO-format dataset. images_dir + annotations JSON."""
  def __init__(self, images_dir, ann_file, img_size=640, training=True):
    self.images_dir = images_dir
    self.img_size = img_size
    self.training = training
    with open(ann_file) as f:
      coco = json.load(f)
    # build id→filename map
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    id2wh   = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    # build category id → contiguous index
    cats = sorted(set(a["category_id"] for a in coco["annotations"]))
    self.cat2idx = {c: i for i, c in enumerate(cats)}
    self.num_classes = len(cats)
    # group annotations by image
    ann_by_img = {}
    for a in coco["annotations"]:
      ann_by_img.setdefault(a["image_id"], []).append(a)
    self.samples = []
    for img_id, fname in id2file.items():
      path = os.path.join(images_dir, fname)
      if not os.path.exists(path): continue
      w, h = id2wh[img_id]
      anns = ann_by_img.get(img_id, [])
      boxes, labels = [], []
      for a in anns:
        x, y, bw, bh = a["bbox"]  # COCO: x,y,w,h in pixels
        cx = (x + bw/2) / w; cy = (y + bh/2) / h
        nw = bw / w; nh = bh / h
        boxes.append([cx, cy, nw, nh])
        labels.append(self.cat2idx[a["category_id"]])
      self.samples.append({
        "path": path,
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64)
      })

  def __len__(self): return len(self.samples)

  def __getitem__(self, idx):
    s = self.samples[idx]
    img = Image.open(s["path"])
    tensor, boxes, labels = augment(img, s["boxes"].copy(), s["labels"].copy(),
                                    self.img_size, self.training)
    return tensor, {"boxes": boxes, "labels": labels}

# ── YOLO Dataset ──────────────────────────────────────────────────────────────

class YOLODataset:
  """Loads YOLOv8-format dataset. images/ and labels/ subdirs with txt files."""
  def __init__(self, data_dir, split="train", img_size=640, training=True):
    self.img_size = img_size; self.training = training
    # parse data.yaml for class count and optional split paths
    yaml_path = os.path.join(data_dir, "data.yaml")
    self.num_classes = 80
    yaml_root = data_dir  # 'path:' field in yaml, if present
    yaml_splits = {}      # e.g. {"train": "images/train2017", "val": "images/val2017"}
    if os.path.exists(yaml_path):
      with open(yaml_path) as f:
        for line in f:
          line = line.rstrip()
          if line.startswith("nc:"):
            try: self.num_classes = int(line.split(":", 1)[1].strip())
            except ValueError: pass
          elif line.startswith("path:"):
            p = line.split(":", 1)[1].strip()
            if os.path.isabs(p):
              yaml_root = p if os.path.isdir(p) else data_dir
            else:
              yaml_root = os.path.join(data_dir, p)
          elif line.startswith(("train:", "val:", "valid:", "test:", "validation:")):
            key, val = line.split(":", 1)
            yaml_splits[key.strip()] = val.strip()
    # resolve split aliases
    split_aliases = {
      "train": ["train"],
      "val":   ["val", "valid", "validation"],
      "test":  ["test"],
    }
    img_dir = None
    for alias in split_aliases.get(split, [split]):
      if alias in yaml_splits:
        rel = yaml_splits[alias]
        candidate = rel if os.path.isabs(rel) else os.path.join(yaml_root, rel)
        if os.path.isdir(candidate):
          img_dir = candidate; break
    # fallback: conventional {data_dir}/{split}/images or {data_dir}/{split}
    if img_dir is None:
      for candidate in [
        os.path.join(data_dir, split, "images"),
        os.path.join(data_dir, split),
      ]:
        if os.path.isdir(candidate): img_dir = candidate; break
    if img_dir is None:
      raise FileNotFoundError(f"Could not find images for split '{split}' in {data_dir}")
    # derive labels dir: images/train2017 → labels/train2017, or sibling labels/
    lbl_dir = img_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    if lbl_dir == img_dir:  # no replacement happened
      lbl_dir = os.path.join(os.path.dirname(img_dir), "labels",
                             os.path.basename(img_dir))
    self.samples = []
    for fname in sorted(os.listdir(img_dir)):
      if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
      stem = os.path.splitext(fname)[0]
      lbl_path = os.path.join(lbl_dir, stem + ".txt")
      boxes, labels = [], []
      if os.path.exists(lbl_path):
        for line in open(lbl_path):
          parts = line.strip().split()
          if len(parts) == 5:
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append([cx, cy, w, h]); labels.append(cls)
      self.samples.append({
        "path": os.path.join(img_dir, fname),
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64)
      })

  def __len__(self): return len(self.samples)

  def __getitem__(self, idx):
    s = self.samples[idx]
    img = Image.open(s["path"])
    tensor, boxes, labels = augment(img, s["boxes"].copy(), s["labels"].copy(),
                                    self.img_size, self.training)
    return tensor, {"boxes": boxes, "labels": labels}

# ── DataLoader ────────────────────────────────────────────────────────────────

def collate(batch):
  """Collate list of (tensor, target) into batched tensors."""
  imgs = np.stack([b[0] for b in batch], axis=0)  # (B,3,H,W)
  targets = [b[1] for b in batch]
  return Tensor(imgs), targets

def make_loader(dataset, batch_size, shuffle=True):
  """Simple generator-based DataLoader (no multiprocessing)."""
  indices = list(range(len(dataset)))
  while True:
    if shuffle: random.shuffle(indices)
    for start in range(0, len(indices)-batch_size+1, batch_size):
      batch = [dataset[indices[start+i]] for i in range(batch_size)]
      yield collate(batch)

def load_dataset(data_path, split="train", img_size=640, training=True):
  """Auto-detect COCO JSON or YOLO format."""
  # YOLO: has data.yaml, or conventional {split}/images/ dir
  yaml = os.path.join(data_path, "data.yaml")
  split_img_dir = os.path.join(data_path, split, "images")
  if os.path.exists(yaml) or os.path.isdir(split_img_dir):
    return YOLODataset(data_path, split, img_size, training)
  # COCO: look for annotations json
  for candidate in ["annotations/instances_train2017.json", "train.json", f"{split}.json",
                    "annotations/train.json", "_annotations.coco.json"]:
    ann = os.path.join(data_path, candidate)
    if os.path.exists(ann):
      img_dir = os.path.join(data_path, split if os.path.exists(os.path.join(data_path, split)) else "")
      return COCODataset(img_dir if os.path.isdir(img_dir) else data_path, ann, img_size, training)
  raise FileNotFoundError(f"Could not detect dataset format in {data_path}")
