# tinyrunner

Train **RF-DETR** on custom datasets — no PyTorch, no ultralytics, just [tinygrad](https://github.com/tinygrad/tinygrad).

```
python train.py --data ./my_dataset --epochs 50 --batch 4
```

---

## What is it?

tinyrunner is a minimal implementation of [RF-DETR](https://blog.roboflow.com/rf-detr/) (Real-time Feature Detection Transformer) built on tinygrad. It covers the full pipeline:

- ResNet-50 backbone (pretrained weights auto-downloaded)
- HybridEncoder: AIFI self-attention + CCFM cross-scale fusion
- Transformer decoder with learned queries
- Hungarian matching loss (focal + L1 + GIoU)
- Cosine LR schedule with warmup
- YOLO and COCO dataset formats

**Why tinygrad?** It runs without PyTorch or CUDA drivers. The clang backend JIT-compiles kernels natively, meaning you can train on CPU without any ML framework overhead.

---

## Architecture

```
Input image  (B, 3, H, W)
      │
      ▼
 ┌──────────┐
 │ ResNet50 │  pretrained backbone
 └──────────┘
      │  C3: (B, 512,  H/8,  W/8 )
      │  C4: (B, 1024, H/16, W/16)
      │  C5: (B, 2048, H/32, W/32)
      ▼
 ┌───────────────┐
 │ HybridEncoder │
 │               │  AIFI: transformer encoder on C5
 │               │  CCFM: top-down + bottom-up FPN fusion
 └───────────────┘
      │  f3: (B, 256, H/8,  W/8 )   ← high res, semantic
      │  g4: (B, 256, H/16, W/16)
      │  g5: (B, 256, H/32, W/32)   ← low res, strong semantics
      │  (flattened + positional encoding → memory tokens)
      ▼
 ┌────────────────────┐
 │ TransformerDecoder │  6 layers, 300 learned queries
 │   self-attention   │  queries attend to each other
 │   cross-attention  │  queries attend to memory
 └────────────────────┘
      │  hs: (B, 300, 256)
      ▼
 ┌──────────┐   ┌──────────┐
 │  MLP ×3  │   │  Linear  │
 │ box_head │   │ cls_head │
 └──────────┘   └──────────┘
      │                │
  boxes (B, 300, 4)   logits (B, 300, C)
  cxcywh [0,1]        raw logits
```

**Training loss** uses Hungarian matching to assign each ground-truth box to the best-matching query, then computes:

| Component       | Weight | Formula                    |
|-----------------|--------|----------------------------|
| Focal loss      |  2.0   | Binary focal (α=0.25, γ=2) |
| L1 box loss     |  5.0   | Mean L1 on matched pairs   |
| GIoU loss       |  2.0   | 1 − GIoU on matched pairs  |

---

## Installation

### Option A: Docker (recommended)

```bash
docker build -t tinyrunner:dev .
docker run --rm -v $(pwd)/data:/data -v $(pwd)/runs:/runs \
  tinyrunner:dev --data /data/my_dataset --epochs 50
```

### Option B: Local (requires clang)

```bash
pip install -r requirements.txt
python train.py --data ./my_dataset --epochs 50
```

`requirements.txt`:
```
tinygrad>=0.12.0
numpy
pillow
```

> **Note:** tinygrad uses clang for JIT compilation by default. First run compiles all kernels (takes ~10–30 min on CPU). Subsequent runs reuse the cache at `~/.cache/tinygrad/`.

---

## Quick Start

### Train on a YOLO dataset

```bash
python train.py \
  --data ./my_yolo_dataset \
  --epochs 50 \
  --batch 4 \
  --lr 1e-4 \
  --img-size 640
```

### Train + evaluate mAP each epoch

```bash
python train.py --data ./dataset --epochs 50 --eval-map
```

### Run the benchmark demo

```bash
# Auto-generates a synthetic dataset for a smoke-test:
python demo.py --data auto --epochs 2 --img-size 64 --batch 2

# Your own dataset:
python demo.py --data ./my_dataset --epochs 20 --batch 4

# Also compare against ultralytics YOLOv8n:
python demo.py --data ./my_dataset --epochs 20 --compare-ultralytics
```

### Resume from a checkpoint

```bash
python train.py --data ./dataset --weights runs/rfdetr/last.safetensors --epochs 100
```

---

## Dataset formats

### YOLO format (recommended)

```
my_dataset/
├── data.yaml          # nc: 5 / names: [cat, dog, ...]
├── train/
│   ├── images/        # .jpg or .png
│   └── labels/        # one .txt per image
└── valid/
    ├── images/
    └── labels/
```

Label files: one detection per line, space-separated `class cx cy w h` (all values normalized to [0, 1]):

```
0 0.512 0.348 0.230 0.415
1 0.821 0.601 0.112 0.089
```

`data.yaml`:
```yaml
nc: 2
names: ['cat', 'dog']
```

### COCO format

```
my_dataset/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Pass the root directory and the split will be auto-detected.

---

## CLI reference

### `train.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | Path to dataset |
| `--epochs` | 50 | Training epochs |
| `--batch` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate (head) |
| `--img-size` | 640 | Input image size (square) |
| `--num-classes` | auto | Override class count |
| `--num-queries` | 300 | Detection queries |
| `--dec-layers` | 6 | Decoder transformer layers |
| `--d-model` | 256 | Model hidden dimension |
| `--save-dir` | `runs/rfdetr` | Output directory |
| `--weights` | — | Resume from checkpoint |
| `--pretrained` | true | Use pretrained ResNet-50 |
| `--no-pretrained` | — | Train backbone from scratch |
| `--val-split` | `valid` | Validation split name |
| `--eval-map` | false | Compute mAP@0.5 each epoch |
| `--device` | auto | Backend: `CLANG`, `CUDA`, `PYTHON` |

### `demo.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `auto` | Dataset path, or `auto` for synthetic |
| `--epochs` | 20 | Training epochs |
| `--compare-ultralytics` | false | Also train YOLOv8n for comparison |

---

## Benchmarks

Tinyrunner is a research implementation of RF-DETR. Published results on COCO val2017:

| Model | mAP@0.5 | mAP@[.5:.95] | Params | FPS (A100) |
|-------|---------|--------------|--------|------------|
| **RF-DETR-B** (paper) | ~70.4 | 53.4 | 29M | 60 |
| **RF-DETR-L** (paper) | ~77.5 | 62.1 | 128M | 23 |
| RT-DETR-R50 (paper)   | ~69.6 | 53.1 | 42M | 108 |
| YOLOv8n (ultralytics) | ~52.9 | 37.3 | 3.2M | — |
| YOLOv8s (ultralytics) | ~61.8 | 44.9 | 11.2M | — |
| YOLOv8l (ultralytics) | ~70.1 | 52.9 | 43.7M | — |

> tinyrunner uses the same RF-DETR-B architecture (ResNet-50 + HybridEncoder, d=256, 6 decoder layers, 300 queries). On COCO with full training and pretrained backbone, results should approach RF-DETR-B.

---

## How training works

### Two-pass gradient strategy

tinygrad evaluates lazily: tensors are only computed when `.numpy()` is called. Hungarian matching requires numpy arrays, so a naive implementation breaks the gradient chain.

tinyrunner uses a **two-pass** approach per batch:

```python
# Pass 1 — eval mode: realize predictions for matching
with Tensor.train(False):
    boxes_m, logits_m = model(imgs)
pb, pl = boxes_m.numpy(), logits_m.numpy()   # triggers computation
matches = criterion.compute_matches(pb, pl, targets)

# Pass 2 — train mode: fresh computation, no .numpy() before backward
with Tensor.train():
    boxes, logits = model(imgs)
    loss, sub = criterion(boxes, logits, targets, matches=matches, match_pb=pb)
    loss.backward()
    opt.step()

# Realize sub-losses only AFTER backward
loss_val = float(loss.numpy())
```

This ensures the full gradient graph is intact for all parameters.

### Learning rate schedule

```
LR
 ^
lr │     ╭─────╮
   │    ╱       ╲
   │   ╱         ╲___
   │──╱               ──╴ 0
   └──────────────────────→ epoch
      warmup    cosine decay
```

Backbone LR = 0.1 × head LR throughout.

---

## Output files

```
runs/rfdetr/
├── best.safetensors   # Best weights by val loss (or mAP if --eval-map)
└── last.safetensors   # Latest weights
```

Load for inference:

```python
from tinyrunner.model import RFDETR
from tinyrunner.eval import postprocess
import numpy as np
from PIL import Image
from tinygrad import Tensor

model = RFDETR(num_classes=2)
model.load("runs/rfdetr/best.safetensors")

img = Image.open("test.jpg").resize((640, 640)).convert("RGB")
x = Tensor(np.array(img).transpose(2,0,1)[None] / 255.0)

with Tensor.train(False):
    boxes, logits = model(x)

b, s, l = postprocess(boxes.numpy()[0], logits.numpy()[0],
                       img_h=640, img_w=640, conf_threshold=0.3)
print(f"{len(b)} detections: {list(zip(s.round(2), l))}")
```

---

## Limitations & roadmap

- **Slow first run**: Kernel compilation takes 10–30 min on CPU (cached after first run)
- **No multi-GPU**: Single device only
- **No EMA**: Exponential moving average weights not yet implemented
- **No augmentation beyond flip**: No mosaic, mixup, colour jitter
- **mAP uses VOC 11-point interpolation**: not identical to COCO's 101-point

Planned:
- [ ] EMA weights
- [ ] More augmentations (mosaic, colour jitter)
- [ ] CUDA backend testing
- [ ] ONNX export
