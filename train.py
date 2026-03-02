#!/usr/bin/env python3
"""
tinyrunner - train RF-DETR on a custom dataset

Usage:
  python train.py --data /path/to/dataset --epochs 50 --batch 4

Dataset formats:
  - YOLO:  directory with data.yaml + train/images/, train/labels/
  - COCO:  directory with images + annotations JSON

Examples:
  python train.py --data ./dataset --epochs 100 --batch 8 --lr 1e-4
  python train.py --data ./coco    --epochs 50  --batch 4 --img-size 640 --pretrained
"""
import argparse, os, sys

def parse_args():
  p = argparse.ArgumentParser(description="Train RF-DETR with tinygrad")
  p.add_argument("--data",        required=True,  help="Path to dataset")
  p.add_argument("--epochs",      type=int,   default=50)
  p.add_argument("--batch",       type=int,   default=4)
  p.add_argument("--lr",          type=float, default=1e-4)
  p.add_argument("--img-size",    type=int,   default=640)
  p.add_argument("--num-classes", type=int,   default=None, help="Override class count")
  p.add_argument("--num-queries", type=int,   default=300)
  p.add_argument("--dec-layers",  type=int,   default=6)
  p.add_argument("--d-model",     type=int,   default=256)
  p.add_argument("--save-dir",    default="runs/rfdetr")
  p.add_argument("--weights",     default=None, help="Load weights from .safetensors")
  p.add_argument("--pretrained",  action="store_true", default=True)
  p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
  p.add_argument("--val-split",   default="valid", help="Validation split name (YOLO format)")
  p.add_argument("--device",      default=None,
                 help="Backend device (e.g. CLANG, CUDA, PYTHON). Auto-detected if not set.")
  return p.parse_args()

def main():
  args = parse_args()

  # Set device before importing tinygrad
  if args.device:
    os.environ[args.device.upper()] = "1"

  from tinyrunner.data import load_dataset, make_loader
  from tinyrunner.model import RFDETR
  from tinyrunner.loss import SetCriterion
  from tinyrunner.trainer import Trainer

  print(f"Loading dataset from {args.data}...")
  train_ds = load_dataset(args.data, split="train", img_size=args.img_size, training=True)
  print(f"  train: {len(train_ds)} images, {train_ds.num_classes} classes")

  val_ds = None
  try:
    val_ds = load_dataset(args.data, split=args.val_split, img_size=args.img_size, training=False)
    print(f"  val:   {len(val_ds)} images")
  except Exception:
    print(f"  (no validation split found at '{args.val_split}')")

  num_classes = args.num_classes or train_ds.num_classes
  print(f"Building RF-DETR: {num_classes} classes, {args.num_queries} queries, d={args.d_model}")

  model = RFDETR(
    num_classes=num_classes,
    num_queries=args.num_queries,
    d=args.d_model,
    dec_layers=args.dec_layers,
  )

  if args.weights:
    model.load(args.weights)

  criterion = SetCriterion(num_classes)

  trainer = Trainer(
    model=model,
    criterion=criterion,
    train_ds=train_ds,
    val_ds=val_ds,
    lr=args.lr,
    batch_size=args.batch,
    epochs=args.epochs,
    img_size=args.img_size,
    save_dir=args.save_dir,
    pretrained_backbone=args.pretrained,
  )

  trainer.train()
  print(f"\nDone. Weights saved to {args.save_dir}/")

if __name__ == "__main__":
  main()
