# TinyRunner - Session Continuation

## Goal
Build a lightweight alternative to ultralytics using tinygrad. RF-DETR model that trains on custom datasets in a single command.

## Current Status: IN PROGRESS — Core code written, need to finish testing + debugging

---

## What's Been Done

### Environment
- **Python venv**: `/home/devuser/.venv/bin/python3` — has numpy, pillow, tinygrad 0.12.0 installed
- **No clang on host** → use Docker for all execution
- **Docker image**: `tinyrunner:dev` — Ubuntu 24.04 + clang + numpy + pillow + tinygrad
- **Notifications**: POST to `http://localhost:8151/notify` with `{"secret":"217cabab-791c-4432-b6ba-474fa1c08718","text":"..."}`
- **uv binary**: `/tmp/uv-x86_64-unknown-linux-gnu/uv` (install packages: `PATH="/tmp/uv-x86_64-unknown-linux-gnu:..." uv pip install ...`)

### Files Created
```
/home/devuser/tinyrunner/
├── tinyrunner/
│   ├── __init__.py          ✅ done
│   ├── model.py             ✅ done — RF-DETR (ResNet50+HybridEncoder+Decoder)
│   ├── loss.py              ✅ done — Hungarian matching + focal/L1/GIoU losses
│   ├── data.py              ✅ done — COCO JSON + YOLO format loaders
│   ├── trainer.py           ✅ done — training loop with cosine LR schedule
│   └── notify.py            ✅ done — nanoclaw Telegram alerts
├── train.py                 ✅ done — CLI entry point
├── Dockerfile               ✅ done
└── requirements.txt         ✅ done
```

### Architecture: RF-DETR
- **Backbone**: ResNet50 (returns C3/C4/C5 features at strides 8/16/32)
- **Encoder**: HybridEncoder — AIFI (transformer on C5) + CCFM (top-down + bottom-up fusion)
- **Decoder**: TransformerDecoder — 300 learnable queries × 6 cross-attention layers
- **Heads**: MLP → 4-d boxes (sigmoid), Linear → class logits (focal loss)
- **Loss**: Hungarian matching + focal classification + L1 box + GIoU

---

## What Needs to Be Done Next

### 1. Fix & Test Loss (PRIORITY)
The `binary_crossentropy_with_logits` bug was fixed → `binary_crossentropy_logits`.
Still need to verify the full loss computation runs without errors.

**Test command:**
```bash
docker run --rm --entrypoint python3 -v /home/devuser/tinyrunner:/app tinyrunner:dev -c "
import sys, numpy as np; sys.path.insert(0, '/app')
from tinyrunner.loss import SetCriterion, _hungarian
from tinygrad import Tensor

# Test Hungarian
cost = np.array([[4,1,3],[2,0,5],[3,2,2]], dtype=float)
rows, cols = _hungarian(cost)
print('Hungarian:', rows, cols)  # should be [2,0,1], [0,1,2] or similar optimal assignment

# Test criterion
criterion = SetCriterion(num_classes=10)
pred_boxes  = Tensor.rand(2, 20, 4)
pred_logits = Tensor.randn(2, 20, 10)
targets = [
  {'boxes': np.array([[0.5,0.5,0.3,0.3]], dtype=np.float32), 'labels': np.array([0], dtype=np.int64)},
  {'boxes': np.array([[0.7,0.6,0.2,0.2]], dtype=np.float32), 'labels': np.array([5], dtype=np.int64)},
]
loss, sub = criterion(pred_boxes, pred_logits, targets)
print('loss:', loss.numpy())
print('sub:', sub)
"
```

### 2. Test Full Model Forward + Loss + Backward
```bash
docker run --rm --entrypoint python3 -v /home/devuser/tinyrunner:/app tinyrunner:dev -c "
import sys, numpy as np; sys.path.insert(0, '/app')
from tinyrunner.model import RFDETR
from tinyrunner.loss import SetCriterion
from tinygrad import Tensor, nn

model = RFDETR(num_classes=3, num_queries=50, d=64, dec_layers=2, heads=4, ffn_d=128)
criterion = SetCriterion(num_classes=3)
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-4)

with Tensor.train():
  opt.zero_grad()
  x = Tensor.randn(1, 3, 160, 160)
  boxes, logits = model(x)
  targets = [{'boxes': np.array([[0.5,0.5,0.3,0.3]], dtype=np.float32), 'labels': np.array([0], dtype=np.int64)}]
  loss, sub = criterion(boxes, logits, targets)
  loss.backward()
  opt.step()
print('One training step OK! loss=', loss.numpy())
"
```

### 3. Known Issues to Check
- [ ] `giou_np` function has a bug — union calculation is wrong (see `loss.py` line ~35). Should use:
  ```python
  def giou_np(b1, b2):
    iou = np.diag(box_iou_np(b1, b2))
    enc_x1 = np.minimum(b1[:,0], b2[:,0]); enc_y1 = np.minimum(b1[:,1], b2[:,1])
    enc_x2 = np.maximum(b1[:,2], b2[:,2]); enc_y2 = np.maximum(b1[:,3], b2[:,3])
    enc_area = np.maximum(0, enc_x2-enc_x1) * np.maximum(0, enc_y2-enc_y1)
    area1 = (b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1])
    area2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    union = area1 + area2 - iou * (area1 + area2 - (area1 + area2 - iou*(area1+area2)))  # broken!
    # CORRECT VERSION:
    inter = iou * (area1 + area2) / (1 + iou)  # not right either
    # Use proper formula: iou = inter/union => union = inter/iou, inter = area1+area2-union
    # Actually: union = (area1 + area2) / (1 + iou) is WRONG
    # CORRECT: compute inter directly from iou and union
    # inter = iou * (area1 + area2 - inter) => inter*(1+iou) = iou*(area1+area2)
    # => inter = iou*(area1+area2)/(1+iou)
    # But this is getting complicated. Simplest correct version:
    ...
  ```
  **Fix**: Replace giou_np with a correct implementation:
  ```python
  def giou_np(b1, b2):
    """GIoU for matched pairs. b1,b2: (N,4) xyxy. Returns (N,)"""
    inter_x1 = np.maximum(b1[:,0], b2[:,0]); inter_y1 = np.maximum(b1[:,1], b2[:,1])
    inter_x2 = np.minimum(b1[:,2], b2[:,2]); inter_y2 = np.minimum(b1[:,3], b2[:,3])
    inter = np.maximum(0, inter_x2-inter_x1) * np.maximum(0, inter_y2-inter_y1)
    area1 = (b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1])
    area2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    enc_x1 = np.minimum(b1[:,0], b2[:,0]); enc_y1 = np.minimum(b1[:,1], b2[:,1])
    enc_x2 = np.maximum(b1[:,2], b2[:,2]); enc_y2 = np.maximum(b1[:,3], b2[:,3])
    enc_area = np.maximum(0, enc_x2-enc_x1) * np.maximum(0, enc_y2-enc_y1)
    return iou - (enc_area - union) / (enc_area + 1e-6)
  ```

- [ ] In `model.py` the `sincos2d` positional encoding may have issues with tensor operations (mixing Python lists and Tensors). Verify it works or simplify.

- [ ] The `Trainer._set_lr` calls `self.opt_head.lr.assign(Tensor(head_lr))` — check if `lr` is a Tensor attribute on AdamW (it should be, but verify)

- [ ] In `train.py` there's `args.d_model` but argparse sets `args.d_model` (underscore, not hyphen-separated). Check argparse dest.

### 4. End-to-End Training Test
Create a tiny dummy dataset and test a full training run:
```bash
# Create dummy YOLO dataset
python3 -c "
import os, numpy as np
from PIL import Image
os.makedirs('/tmp/ds/train/images', exist_ok=True)
os.makedirs('/tmp/ds/train/labels', exist_ok=True)
for i in range(8):
  img = Image.fromarray(np.random.randint(0,255,(160,160,3),dtype=np.uint8))
  img.save(f'/tmp/ds/train/images/img{i:04d}.jpg')
  with open(f'/tmp/ds/train/labels/img{i:04d}.txt','w') as f:
    f.write('0 0.5 0.5 0.3 0.3\n1 0.2 0.3 0.1 0.1\n')
with open('/tmp/ds/data.yaml','w') as f: f.write('nc: 2\nnames: [cat, dog]\n')
print('Dataset created')
"

# Run training
docker run --rm --entrypoint python3 -v /home/devuser/tinyrunner:/app -v /tmp/ds:/data tinyrunner:dev /app/train.py \
  --data /data --epochs 3 --batch 2 --img-size 160 --num-queries 50 --d-model 64 --dec-layers 2 --no-pretrained
```

### 5. After Tests Pass
- Send notification: `curl -s -X POST http://localhost:8151/notify -H 'Content-Type: application/json' -d '{"secret":"217cabab-791c-4432-b6ba-474fa1c08718","text":"RF-DETR core tests passed!"}'`
- Fix any remaining issues
- Add EMA (Exponential Moving Average) for model weights (improves final mAP)
- Add mAP validation metric using COCO evaluator (numpy-based)
- Add TinyJit for faster training
- Polish CLI help text and README

---

## Known Tinygrad Gotchas (0.12.0)
- `Tensor.test()` does NOT exist — use `Tensor.train(False)` or just no context manager
- `binary_crossentropy_with_logits` → `binary_crossentropy_logits`
- `AdamW` takes a flat list of params (no param groups) — use `OptimizerGroup` for different LRs
- Default device auto-detects; if clang not available, set env var `PYTHON=1` (very slow) or run in Docker
- `Tensor.expand(-1, ...)` works (preserves that dim)
- Use `nn.state.get_state_dict(model)` to get all named parameters
- Optimizer `lr` is a `Tensor` attribute that can be `.assign()`-ed

## How to Build/Run
```bash
# Build docker image
cd /home/devuser/tinyrunner
docker build -t tinyrunner:dev .

# Run training (YOLO format)
docker run --rm -v /path/to/dataset:/data -v $(pwd):/app tinyrunner:dev \
  --entrypoint python3 /app/train.py --data /data --epochs 50 --batch 8

# Or run with GPU (when CUDA available)
docker run --rm --gpus all -v /path/to/dataset:/data tinyrunner:dev \
  --data /data --device CUDA --epochs 50 --batch 8
```

## File Quick Reference
| File | Purpose | Key classes/functions |
|------|---------|----------------------|
| `tinyrunner/model.py` | RF-DETR model | `RFDETR`, `ResNet50`, `HybridEncoder`, `TransformerDecoder` |
| `tinyrunner/loss.py` | Loss functions | `SetCriterion`, `_hungarian`, `match` |
| `tinyrunner/data.py` | Data loading | `COCODataset`, `YOLODataset`, `load_dataset` |
| `tinyrunner/trainer.py` | Training loop | `Trainer` |
| `tinyrunner/notify.py` | Notifications | `notify(text)` |
| `train.py` | CLI entry | `main()` |
