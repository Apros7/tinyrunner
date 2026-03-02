#!/usr/bin/env python3
"""
tinyrunner demo — train RF-DETR, evaluate mAP, compare against benchmarks.

  uv run demo.py                          # downloads COCO128, auto-detects CUDA
  uv run demo.py --data /my/dataset       # your own YOLO dataset
  uv run demo.py --epochs 50 --batch 8    # more training
"""
import argparse, os, sys, time, subprocess, urllib.request, zipfile, shutil
import numpy as np

# ── CUDA auto-detection (must happen before tinygrad import) ──────────────────
def _detect_device():
  if os.environ.get("CUDA") or os.environ.get("GPU"):
    return "CUDA"
  try:
    r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    if r.returncode == 0:
      os.environ["CUDA"] = "1"
      return "CUDA"
  except (FileNotFoundError, subprocess.TimeoutExpired):
    pass
  return "CLANG"


def _nvrtc_test(libnvrtc, arch):
  """Return True if nvrtc can compile a trivial kernel for arch."""
  import ctypes
  src = b'extern "C" __global__ void k() {}'
  prog = ctypes.c_void_p()
  libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None)
  opt = f"--gpu-architecture={arch}".encode()
  status = libnvrtc.nvrtcCompileProgram(prog, 1, (ctypes.c_char_p * 1)(opt))
  libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
  return status == 0


def _find_nvrtc_fallback():
  """
  Test if nvrtc supports the GPU's native arch.  If not, return the highest
  virtual arch (compute_XX) that nvrtc does support.

  Virtual archs produce forward-compatible PTX that runs on any newer GPU.
  nvrtc compiles these in milliseconds — no slow CUDA-driver JIT needed.

  Returns (gpu_arch, effective_arch) where effective_arch may differ from
  gpu_arch when a fallback is needed, or (None, None) on failure.
  """
  import ctypes
  try:
    for libname in ("libcuda.so.1", "libcuda.so"):
      try: libcuda = ctypes.CDLL(libname); break
      except OSError: pass
    else: return None, None
    if libcuda.cuInit(0) != 0: return None, None
    major, minor = ctypes.c_int(), ctypes.c_int()
    libcuda.cuDeviceGetAttribute(ctypes.byref(major), 75, ctypes.c_int(0))
    libcuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, ctypes.c_int(0))
    gpu_arch = f"sm_{major.value}{minor.value}"
  except Exception:
    return None, None

  try:
    for libname in ("libnvrtc.so.1", "libnvrtc.so.12", "libnvrtc.so"):
      try: libnvrtc = ctypes.CDLL(libname); break
      except OSError: pass
    else: return gpu_arch, None
    if _nvrtc_test(libnvrtc, gpu_arch):
      return gpu_arch, gpu_arch   # nvrtc supports this GPU natively
    # Try virtual archs (forward-compatible PTX, works on any newer GPU)
    for va in ("compute_90", "compute_89", "compute_87", "compute_86", "compute_80", "compute_75"):
      if _nvrtc_test(libnvrtc, va):
        return gpu_arch, va
  except Exception:
    pass
  return gpu_arch, None  # Nothing works → caller falls back to CUDA_PTX


def _patch_cuda_compiler(effective_arch):
  """Monkey-patch CUDACompiler to use effective_arch instead of detected GPU arch."""
  try:
    from tinygrad.runtime.support.compiler_cuda import CUDACompiler
    orig = CUDACompiler.__init__
    def _patched(self, arch, cache_key="cuda"):
      orig(self, effective_arch, cache_key)
    CUDACompiler.__init__ = _patched
    return True
  except Exception:
    return False


def _configure_cuda():
  """Called in main() after tinygrad is imported, before any tensor ops."""
  if os.environ.get("CUDA_PTX") or os.environ.get("CUDA_CC"):
    return  # user already set something
  gpu_arch, effective_arch = _find_nvrtc_fallback()
  if gpu_arch is None:
    return  # couldn't probe, leave defaults
  if effective_arch == gpu_arch:
    print(f"  nvrtc: native {gpu_arch}", flush=True)
    return  # nvrtc works natively, nothing to do
  if effective_arch is not None:
    print(f"  nvrtc: {gpu_arch} unsupported, patching to {effective_arch}", flush=True)
    if not _patch_cuda_compiler(effective_arch):
      print(f"  patch failed, falling back to CUDA_PTX=1 (slow first batch)", flush=True)
      os.environ["CUDA_PTX"] = "1"
  else:
    print(f"  nvrtc: no compatible arch found, using CUDA_PTX=1 (slow first batch)", flush=True)
    os.environ["CUDA_PTX"] = "1"

DEVICE = _detect_device()

# ── Published benchmark reference numbers ─────────────────────────────────────
PUBLISHED = [
  {"model": "RF-DETR-B (paper)",     "mAP@0.5": 70.4, "params": "29M",  "note": "COCO val2017, 640px"},
  {"model": "RF-DETR-L (paper)",     "mAP@0.5": 77.5, "params": "128M", "note": "COCO val2017, 1024px"},
  {"model": "RT-DETR-R50 (paper)",   "mAP@0.5": 69.6, "params": "42M",  "note": "COCO val2017, 640px"},
  {"model": "YOLOv8n (ultralytics)", "mAP@0.5": 52.9, "params": "3.2M", "note": "COCO val2017, 640px"},
  {"model": "YOLOv8s (ultralytics)", "mAP@0.5": 61.8, "params": "11M",  "note": "COCO val2017, 640px"},
  {"model": "YOLOv8l (ultralytics)", "mAP@0.5": 70.1, "params": "44M",  "note": "COCO val2017, 640px"},
]

COCO128_URL  = "https://ultralytics.com/assets/coco128.zip"
COCO128_DIR  = os.path.join(os.path.dirname(__file__), "demo_data", "coco128")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _download_coco128():
  """Download and unpack COCO128 (~7MB) the first time. Returns dataset path."""
  if os.path.isdir(COCO128_DIR):
    return COCO128_DIR
  print("Downloading COCO128 demo dataset (~7MB)...", flush=True)
  os.makedirs(os.path.dirname(COCO128_DIR), exist_ok=True)
  zip_path = COCO128_DIR + ".zip"
  urllib.request.urlretrieve(COCO128_URL, zip_path,
    reporthook=lambda b, bs, total: print(
      f"  {min(b*bs, total)/1e6:.1f}/{total/1e6:.1f} MB\r", end="", flush=True) if total > 0 else None)
  print()
  with zipfile.ZipFile(zip_path) as z:
    z.extractall(os.path.dirname(COCO128_DIR))
  os.rename(os.path.join(os.path.dirname(COCO128_DIR), "coco128"), COCO128_DIR)
  os.remove(zip_path)
  # COCO128 ships its own data.yaml with absolute paths baked in for the
  # ultralytics machine — always overwrite with our actual paths.
  yaml_path = os.path.join(COCO128_DIR, "data.yaml")
  _write_coco128_yaml(yaml_path)
  print(f"  Saved to {COCO128_DIR}", flush=True)
  return COCO128_DIR


def _write_coco128_yaml(yaml_path):
  names = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
           "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
           "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
           "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
           "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
           "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
           "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
           "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
           "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
           "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
           "hair drier","toothbrush"]
  root = os.path.dirname(yaml_path)
  # Find train/val splits
  train_img = None
  for candidate in ["images/train2017", "images/train", "train/images"]:
    if os.path.isdir(os.path.join(root, candidate)):
      train_img = candidate; break
  val_img = None
  for candidate in ["images/val2017", "images/val", "valid/images"]:
    if os.path.isdir(os.path.join(root, candidate)):
      val_img = candidate; break
  with open(yaml_path, "w") as f:
    f.write(f"path: {root}\n")
    if train_img: f.write(f"train: {train_img}\n")
    if val_img:   f.write(f"val: {val_img}\n")
    f.write(f"nc: {len(names)}\n")
    f.write(f"names: {names}\n")


def _find_split(data_path, split_names, img_size=640):
  """Try multiple split directory names, return dataset or None."""
  from tinyrunner.data import load_dataset
  for name in split_names:
    try:
      return load_dataset(data_path, split=name, img_size=img_size, training=False)
    except Exception:
      pass
  return None


# ── Table printing ─────────────────────────────────────────────────────────────

def _print_table(rows, headers):
  widths = [max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))
            for h in headers]
  sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
  fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
  print(sep)
  print(fmt.format(*headers))
  print(sep)
  for r in rows:
    print(fmt.format(*[str(r.get(h, "")) for h in headers]))
  print(sep)


def _param_count(model):
  from tinygrad.nn import state as nn_state
  return sum(int(np.prod(p.shape)) for p in nn_state.get_parameters(model)) / 1e6


# ── Ultralytics comparison ────────────────────────────────────────────────────

def _run_ultralytics(data_path, epochs, img_size, batch, save_dir):
  """Train YOLOv8n on the same data and return mAP@0.5, or None if unavailable."""
  try:
    from ultralytics import YOLO
  except ImportError:
    return None

  yaml_path = os.path.join(data_path, "data.yaml")
  if not os.path.exists(yaml_path):
    return None

  print("\n── YOLOv8n comparison (ultralytics) ────────────────────────────────", flush=True)
  yolo_dir = os.path.join(save_dir, "yolo")
  os.makedirs(yolo_dir, exist_ok=True)
  model = YOLO("yolov8n.pt")
  model.train(data=yaml_path, epochs=epochs, imgsz=img_size, batch=batch,
              project=yolo_dir, name="run", verbose=False)
  val_res = model.val(data=yaml_path, imgsz=img_size, batch=batch, verbose=False)
  mAP = float(val_res.box.map50)
  print(f"  YOLOv8n  mAP@0.5 = {mAP:.4f}", flush=True)
  return mAP


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
  p = argparse.ArgumentParser(
    description="tinyrunner demo: train RF-DETR, evaluate mAP, compare against benchmarks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument("--data",     default=None,
                 help="YOLO dataset path. Omit to auto-download COCO128.")
  p.add_argument("--epochs",   type=int,   default=20)
  p.add_argument("--batch",    type=int,   default=8)
  p.add_argument("--lr",       type=float, default=1e-4)
  p.add_argument("--img-size", type=int,   default=640)
  p.add_argument("--save-dir", default="demo_out")
  p.add_argument("--weights",  default=None, help="Resume from .safetensors checkpoint")
  p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True,
                 help="Skip pretrained ResNet-50 backbone")
  p.add_argument("--device",   default=None,
                 help="Force backend: CLANG, CUDA, PYTHON (auto-detected by default)")
  return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
  args = parse_args()

  # Device — CLI overrides auto-detection
  if args.device:
    os.environ[args.device.upper()] = "1"
    device = args.device.upper()
  else:
    device = DEVICE
  print(f"Device: {device}", flush=True)

  # Dataset
  if args.data:
    data_path = args.data
  else:
    data_path = _download_coco128()

  # Now safe to import tinygrad (device env var is set)
  from tinyrunner.data import load_dataset, make_loader
  from tinyrunner.model import RFDETR
  from tinyrunner.loss import SetCriterion
  if device == "CUDA":
    _configure_cuda()  # patch nvrtc arch if needed, before any tensor ops
  from tinyrunner.trainer import Trainer
  from tinyrunner.eval import evaluate
  from tinyrunner.notify import notify

  # ── Load dataset ────────────────────────────────────────────────────────────
  print(f"\n── Dataset: {data_path}", flush=True)
  train_ds = load_dataset(data_path, split="train", img_size=args.img_size, training=True)
  print(f"  train : {len(train_ds)} images, {train_ds.num_classes} classes")
  val_ds = _find_split(data_path, ["valid", "val", "validation", "test"], img_size=args.img_size)
  if val_ds:
    print(f"  val   : {len(val_ds)} images")
  else:
    print("  val   : not found — will skip evaluation")

  num_classes = train_ds.num_classes

  # ── Build model ─────────────────────────────────────────────────────────────
  print(f"\n── RF-DETR model", flush=True)
  model = RFDETR(num_classes=num_classes)
  if args.weights:
    model.load(args.weights)
    print(f"  Loaded: {args.weights}")
  params_M = _param_count(model)
  print(f"  Params  : {params_M:.1f}M")
  print(f"  Classes : {num_classes}")
  print(f"  Queries : 300")

  criterion = SetCriterion(num_classes)
  os.makedirs(args.save_dir, exist_ok=True)

  trainer = Trainer(
    model=model, criterion=criterion,
    train_ds=train_ds, val_ds=val_ds,
    lr=args.lr, batch_size=args.batch,
    epochs=args.epochs, img_size=args.img_size,
    save_dir=args.save_dir,
    pretrained_backbone=args.pretrained,
  )

  # ── Train ───────────────────────────────────────────────────────────────────
  print(f"\n── Training ({args.epochs} epochs, batch={args.batch}, lr={args.lr})", flush=True)
  t0 = time.time()
  trainer.train()
  train_time = time.time() - t0
  print(f"\nTraining: {train_time:.0f}s total  ({train_time/args.epochs:.0f}s/epoch)", flush=True)

  # ── Evaluate tinyrunner ──────────────────────────────────────────────────────
  tinyrunner_map = None
  if val_ds is not None:
    print(f"\n── Evaluating mAP@0.5", flush=True)
    best_path = os.path.join(args.save_dir, "best.safetensors")
    if os.path.exists(best_path):
      model.load(best_path)
    res = evaluate(model, val_ds, num_classes=num_classes,
                   img_size=args.img_size, batch_size=args.batch)
    tinyrunner_map = res["mAP"]
    print(f"  tinyrunner mAP@0.5 = {tinyrunner_map:.4f}")
    class_names = getattr(train_ds, "class_names", None) or {i: f"cls{i}" for i in range(num_classes)}
    for c, ap in res["AP"].items():
      if res["n_gt"].get(c, 0) > 0:
        name = (class_names[c] if isinstance(class_names, (list, dict))
                and c < len(class_names) else f"cls{c}")
        print(f"    {str(name):20s}  AP={ap:.4f}  GT={res['n_gt'][c]}")

  # ── Ultralytics comparison (automatic if installed) ─────────────────────────
  ultralytics_map = _run_ultralytics(data_path, args.epochs, args.img_size, args.batch, args.save_dir)

  # ── Results table ────────────────────────────────────────────────────────────
  print(f"\n── Results", flush=True)
  rows = []
  if tinyrunner_map is not None:
    rows.append({
      "Model": f"tinyrunner RF-DETR ({args.epochs} epochs)",
      "mAP@0.5": f"{tinyrunner_map:.4f}",
      "Params": f"{params_M:.1f}M",
      "Dataset": f"{len(train_ds)} train imgs",
    })
  if ultralytics_map is not None:
    rows.append({
      "Model": f"YOLOv8n ultralytics ({args.epochs} epochs)",
      "mAP@0.5": f"{ultralytics_map:.4f}",
      "Params": "3.2M",
      "Dataset": f"{len(train_ds)} train imgs",
    })
  rows.append({"Model": "── published COCO benchmarks ──", "mAP@0.5": "", "Params": "", "Dataset": ""})
  for r in PUBLISHED:
    rows.append({"Model": r["model"], "mAP@0.5": str(r["mAP@0.5"]), "Params": r["params"], "Dataset": r["note"]})

  _print_table(rows, ["Model", "mAP@0.5", "Params", "Dataset"])

  if tinyrunner_map is not None:
    notify(f"tinyrunner demo: mAP@0.5={tinyrunner_map:.4f}  ({args.epochs}ep, {len(train_ds)} imgs, {device})")

  print(f"\nWeights : {args.save_dir}/best.safetensors")
  print("Done.")


if __name__ == "__main__":
  main()
