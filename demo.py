#!/usr/bin/env python3
"""
tinyrunner demo — train RF-DETR, evaluate mAP, compare against benchmarks.

Usage:
  python demo.py --data /path/to/yolo_dataset [--epochs 20] [--batch 4]
                 [--img-size 640] [--save-dir demo_out]
                 [--compare-ultralytics]

Quick smoke-test with a synthetic dataset:
  python demo.py --data auto --epochs 2 --img-size 64 --batch 2
"""
import argparse, os, sys, time, math, tempfile
import numpy as np

# ── Published benchmark reference numbers ─────────────────────────────────────
# Source: RF-DETR paper (Roboflow, 2024) and Ultralytics docs
PUBLISHED = [
  {"model": "RF-DETR-B (paper)",  "mAP_COCO": 53.4, "params_M": 29,    "note": "COCO val2017, 640px"},
  {"model": "RF-DETR-L (paper)",  "mAP_COCO": 62.1, "params_M": 128,   "note": "COCO val2017, 1024px"},
  {"model": "YOLOv8n (ultralytics)", "mAP_COCO": 37.3, "params_M": 3.2,  "note": "COCO val2017, 640px"},
  {"model": "YOLOv8s (ultralytics)", "mAP_COCO": 44.9, "params_M": 11.2, "note": "COCO val2017, 640px"},
  {"model": "YOLOv8l (ultralytics)", "mAP_COCO": 52.9, "params_M": 43.7, "note": "COCO val2017, 640px"},
  {"model": "RT-DETR-R50 (paper)",   "mAP_COCO": 53.1, "params_M": 42,   "note": "COCO val2017, 640px"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_demo_dataset(root, n_train=16, n_val=8, num_classes=2, img_size=64):
  """Create a tiny synthetic YOLO-format dataset for smoke-testing."""
  from PIL import Image
  for split, n in [("train", n_train), ("valid", n_val)]:
    os.makedirs(f"{root}/{split}/images", exist_ok=True)
    os.makedirs(f"{root}/{split}/labels", exist_ok=True)
  rng = np.random.default_rng(42)
  for split, n in [("train", n_train), ("valid", n_val)]:
    for i in range(n):
      img = rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)
      Image.fromarray(img).save(f"{root}/{split}/images/img{i:04d}.jpg")
      cls = rng.integers(0, num_classes)
      cx, cy = rng.uniform(0.2, 0.8, 2)
      w, h = rng.uniform(0.1, 0.3, 2)
      with open(f"{root}/{split}/labels/img{i:04d}.txt", "w") as f:
        f.write(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
  with open(f"{root}/data.yaml", "w") as f:
    names = [f"class{i}" for i in range(num_classes)]
    f.write(f"nc: {num_classes}\nnames: {names}\n")
  return num_classes


def _print_table(rows, headers):
  widths = [max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))
            for h in headers]
  sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
  fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
  print(sep)
  print(fmt.format(*headers))
  print(sep)
  for r in rows:
    print(fmt.format(*[str(r.get(h, "—")) for h in headers]))
  print(sep)


def _param_count(model):
  from tinygrad.nn import state as nn_state
  return sum(np.prod(p.shape) for p in nn_state.get_parameters(model)) / 1e6


# ── Ultralytics comparison ────────────────────────────────────────────────────

def _run_ultralytics(data_path, num_classes, epochs, img_size, batch, save_dir):
  """Train YOLOv8n on same data and evaluate mAP. Returns mAP float or None."""
  try:
    from ultralytics import YOLO
  except ImportError:
    print("  ultralytics not installed — skipping live comparison")
    return None

  print("\n── Ultralytics YOLOv8n comparison ──────────────────────────────────")
  yaml_path = os.path.join(data_path, "data.yaml")
  if not os.path.exists(yaml_path):
    print("  No data.yaml found — skipping ultralytics comparison")
    return None

  # Patch data.yaml to include absolute paths if needed
  yolo_dir = os.path.join(save_dir, "yolo")
  os.makedirs(yolo_dir, exist_ok=True)

  model = YOLO("yolov8n.pt")
  results = model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=img_size,
    batch=batch,
    project=yolo_dir,
    name="run",
    verbose=False,
  )
  val_results = model.val(data=yaml_path, imgsz=img_size, batch=batch, verbose=False)
  mAP = float(val_results.box.map50)
  print(f"  YOLOv8n  mAP@0.5 = {mAP:.4f}", flush=True)
  return mAP


# ── Main demo ─────────────────────────────────────────────────────────────────

def parse_args():
  p = argparse.ArgumentParser(description="tinyrunner demo: train RF-DETR and benchmark")
  p.add_argument("--data",    default=None,
                 help="Path to YOLO dataset, or 'auto' to generate a synthetic one")
  p.add_argument("--epochs",  type=int,   default=20)
  p.add_argument("--batch",   type=int,   default=4)
  p.add_argument("--lr",      type=float, default=1e-4)
  p.add_argument("--img-size",type=int,   default=640)
  p.add_argument("--save-dir",default="demo_out")
  p.add_argument("--weights", default=None, help="Resume from .safetensors checkpoint")
  p.add_argument("--pretrained", action="store_true", default=True)
  p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
  p.add_argument("--compare-ultralytics", action="store_true", default=False,
                 help="Also train YOLOv8n (requires ultralytics package)")
  p.add_argument("--device",  default=None,
                 help="Backend (CLANG, CUDA, PYTHON). Auto-detected if not set.")
  return p.parse_args()


def main():
  args = parse_args()

  if args.device:
    os.environ[args.device.upper()] = "1"

  # ── Dataset setup ──────────────────────────────────────────────────────────
  tmpdir = None
  if args.data is None or args.data.lower() == "auto":
    print("No dataset provided — generating synthetic demo dataset...")
    tmpdir = tempfile.mkdtemp(prefix="tinyrunner_demo_")
    num_classes = _make_demo_dataset(tmpdir, img_size=min(args.img_size, 64))
    data_path = tmpdir
    print(f"  Synthetic dataset at {tmpdir}  ({num_classes} classes, 16 train / 8 val)")
  else:
    data_path = args.data

  from tinyrunner.data import load_dataset
  from tinyrunner.model import RFDETR
  from tinyrunner.loss import SetCriterion
  from tinyrunner.trainer import Trainer
  from tinyrunner.eval import evaluate
  from tinyrunner.notify import notify

  print(f"\n── Dataset ─────────────────────────────────────────────────────────")
  train_ds = load_dataset(data_path, split="train", img_size=args.img_size, training=True)
  print(f"  train: {len(train_ds)} images, {train_ds.num_classes} classes")
  val_ds = None
  try:
    val_ds = load_dataset(data_path, split="valid", img_size=args.img_size, training=False)
    print(f"  val:   {len(val_ds)} images")
  except Exception:
    try:
      val_ds = load_dataset(data_path, split="val", img_size=args.img_size, training=False)
      print(f"  val:   {len(val_ds)} images")
    except Exception:
      print("  (no validation split found)")

  num_classes = train_ds.num_classes

  # ── Build model ────────────────────────────────────────────────────────────
  print(f"\n── Model ───────────────────────────────────────────────────────────")
  model = RFDETR(num_classes=num_classes)
  if args.weights:
    model.load(args.weights)
    print(f"  Loaded weights from {args.weights}")
  params_M = _param_count(model)
  print(f"  Parameters: {params_M:.1f}M")
  print(f"  Classes:    {num_classes}")
  print(f"  Queries:    300  (default)")

  criterion = SetCriterion(num_classes)
  os.makedirs(args.save_dir, exist_ok=True)

  trainer = Trainer(
    model=model, criterion=criterion,
    train_ds=train_ds, val_ds=val_ds,
    lr=args.lr, batch_size=args.batch,
    epochs=args.epochs, img_size=args.img_size,
    save_dir=args.save_dir,
    pretrained_backbone=args.pretrained,
    eval_map=False,  # We'll compute mAP ourselves at the end
  )

  # ── Train ──────────────────────────────────────────────────────────────────
  print(f"\n── Training RF-DETR ({args.epochs} epochs) ──────────────────────────")
  t_train = time.time()
  trainer.train()
  train_time = time.time() - t_train
  print(f"\nTraining took {train_time:.0f}s ({train_time/args.epochs:.0f}s/epoch)", flush=True)

  # ── Evaluate tinyrunner mAP ────────────────────────────────────────────────
  tinyrunner_map = None
  if val_ds is not None:
    print(f"\n── Evaluating tinyrunner mAP@0.5 ───────────────────────────────────")
    # Load best weights for evaluation
    best_path = os.path.join(args.save_dir, "best.safetensors")
    if os.path.exists(best_path):
      model.load(best_path)
      print(f"  Using best weights: {best_path}")
    res = evaluate(model, val_ds, num_classes=num_classes,
                   img_size=args.img_size, batch_size=args.batch)
    tinyrunner_map = res["mAP"]
    print(f"  mAP@0.5 = {tinyrunner_map:.4f}")
    if res["AP"]:
      print("  Per-class AP:")
      class_names = getattr(train_ds, "class_names",
                            {i: f"class{i}" for i in range(num_classes)})
      for c, ap in res["AP"].items():
        if res["n_gt"].get(c, 0) > 0:
          name = class_names.get(c, f"class{c}") if isinstance(class_names, dict) else (
            class_names[c] if c < len(class_names) else f"class{c}")
          print(f"    {name:20s}  AP={ap:.4f}  (GT={res['n_gt'][c]}, pred={res['n_pred'][c]})")

  # ── Ultralytics comparison ─────────────────────────────────────────────────
  ultralytics_map = None
  if args.compare_ultralytics:
    ultralytics_map = _run_ultralytics(
      data_path, num_classes, args.epochs, args.img_size, args.batch, args.save_dir)

  # ── Results table ──────────────────────────────────────────────────────────
  print(f"\n── Results ─────────────────────────────────────────────────────────")
  rows = []
  if tinyrunner_map is not None:
    rows.append({
      "Model": f"tinyrunner RF-DETR (this, {args.epochs}ep)",
      "mAP@0.5 (dataset)": f"{tinyrunner_map:.4f}",
      "Params": f"{params_M:.1f}M",
      "Note": f"{len(train_ds)} train imgs, {args.img_size}px",
    })
  if ultralytics_map is not None:
    rows.append({
      "Model": f"YOLOv8n ultralytics ({args.epochs}ep)",
      "mAP@0.5 (dataset)": f"{ultralytics_map:.4f}",
      "Params": "3.2M",
      "Note": f"{len(train_ds)} train imgs, {args.img_size}px",
    })

  # Published COCO numbers as context
  rows.append({"Model": "— COCO benchmarks (published) —", "mAP@0.5 (dataset)": "", "Params": "", "Note": ""})
  for r in PUBLISHED:
    rows.append({
      "Model": r["model"],
      "mAP@0.5 (dataset)": f"{r['mAP_COCO']}",
      "Params": f"{r['params_M']}M",
      "Note": r["note"],
    })

  _print_table(rows, ["Model", "mAP@0.5 (dataset)", "Params", "Note"])

  if tinyrunner_map is not None:
    msg = (f"tinyrunner demo done: mAP@0.5={tinyrunner_map:.4f} "
           f"({args.epochs}ep, {len(train_ds)} imgs)")
    notify(msg)

  # Cleanup synthetic dataset
  if tmpdir:
    import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

  print(f"\nWeights saved to: {args.save_dir}/")
  print("Done.")


if __name__ == "__main__":
  main()
