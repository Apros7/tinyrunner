"""Trainer for RF-DETR."""
import time, os, math
from tinygrad import Tensor, nn
from tinygrad.nn import state as nn_state
from tqdm import tqdm
from .notify import notify


def _fmt_time(seconds):
  """Format seconds as m:ss or h:mm:ss."""
  seconds = int(seconds)
  h, rem = divmod(seconds, 3600)
  m, s = divmod(rem, 60)
  return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

class Trainer:
  def __init__(self, model, criterion, train_ds, val_ds=None,
               lr=1e-4, batch_size=4, epochs=50, img_size=640,
               save_dir="runs", pretrained_backbone=True, warmup_epochs=2,
               eval_map=False):
    self.model = model; self.criterion = criterion
    self.train_ds = train_ds; self.val_ds = val_ds
    self.epochs = epochs; self.batch_size = batch_size
    self.save_dir = save_dir; self.lr = lr; self.img_size = img_size
    self.eval_map = eval_map
    os.makedirs(save_dir, exist_ok=True)

    # Separate backbone (lower LR) and head parameters
    backbone_p = nn_state.get_parameters(model.backbone)
    head_p     = (nn_state.get_parameters(model.encoder) + nn_state.get_parameters(model.decoder) +
                  nn_state.get_parameters(model.box_head) + nn_state.get_parameters(model.cls_head))
    self.opt_backbone = nn.optim.AdamW(backbone_p, lr=lr*0.1, weight_decay=1e-4) if backbone_p else None
    self.opt_head     = nn.optim.AdamW(head_p,     lr=lr,     weight_decay=1e-4)

    self.steps_per_epoch = max(1, len(train_ds) // batch_size)
    self.warmup_steps = warmup_epochs * self.steps_per_epoch
    self.total_steps  = epochs * self.steps_per_epoch
    self.step_n = 0
    self.best_loss = float("inf")

    if pretrained_backbone:
      try: model.load_pretrained_backbone()
      except Exception as e: print(f"Warning: pretrained backbone failed: {e}")

  def _set_lr(self, step):
    """Cosine warmup LR schedule."""
    if step <= self.warmup_steps:
      ratio = step / max(1, self.warmup_steps)
    else:
      progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
      ratio = (1 + math.cos(math.pi * progress)) / 2
    head_lr = self.lr * ratio
    if self.opt_backbone: self.opt_backbone.lr.assign(Tensor(head_lr * 0.1))
    self.opt_head.lr.assign(Tensor(head_lr))

  def _step(self):
    self.opt_head.zero_grad()
    if self.opt_backbone: self.opt_backbone.zero_grad()

  def _update(self):
    self.opt_head.step()
    if self.opt_backbone: self.opt_backbone.step()
    self.step_n += 1
    self._set_lr(self.step_n)

  def _epoch(self, dataset, training, epoch_bar=None):
    from .data import make_loader
    loader = make_loader(dataset, self.batch_size, shuffle=training)
    n = len(dataset) // self.batch_size
    total_loss = 0.0; steps = 0
    tag = "train" if training else "val  "
    # Inner step bar — leave=False so it erases itself when done, keeping the epoch bar clean
    step_bar = tqdm(total=n, desc=f"    {tag}", unit="step",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    dynamic_ncols=True, leave=False, position=1)
    t_step = time.time()
    for imgs, targets in loader:
      if steps >= n: break
      if training:
        self._step()
        with Tensor.train(False):
          boxes_m, logits_m = self.model(imgs)
        pb = boxes_m.numpy(); pl = logits_m.numpy()
        matches = self.criterion.compute_matches(pb, pl, targets)
        with Tensor.train():
          boxes, logits = self.model(imgs)
          loss, sub = self.criterion(boxes, logits, targets, matches=matches, match_pb=pb)
          loss.backward()
          self._update()
        loss_val = float(loss.numpy())
        sub_vals = {k: float(v.numpy()) for k, v in sub.items()}
      else:
        with Tensor.train(False):
          boxes, logits = self.model(imgs)
          loss, sub = self.criterion(boxes, logits, targets)
        loss_val = float(loss.numpy())
        sub_vals = {k: float(v.numpy()) for k, v in sub.items()}

      total_loss += loss_val; steps += 1
      step_time = time.time() - t_step; t_step = time.time()
      imgs_per_sec = self.batch_size / max(step_time, 1e-6)
      step_bar.set_postfix(
        loss=f"{total_loss/steps:.4f}",
        cls=f"{sub_vals['cls']:.3f}",
        box=f"{sub_vals['box']:.3f}",
        giou=f"{sub_vals['giou']:.3f}",
        img_s=f"{imgs_per_sec:.1f}",
      )
      step_bar.update(1)
      if epoch_bar is not None:
        epoch_bar.set_postfix(
          loss=f"{total_loss/steps:.4f}",
          img_s=f"{imgs_per_sec:.1f}",
        )
    step_bar.close()
    return total_loss / max(1, steps)

  def train(self):
    notify("RF-DETR training started")
    tqdm.write(f"Training RF-DETR for {self.epochs} epochs, batch={self.batch_size}, lr={self.lr}")
    import os
    if os.environ.get("CUDA_PTX"):
      tqdm.write(
        "  Note: CUDA_PTX=1 — CUDA driver will JIT-compile GPU kernels on the first batch.\n"
        "  This is a one-time cost (~2-5 min for this model) cached in ~/.nv/ComputeCache/.\n"
        "  GPU utilization in nvidia-smi will be 0%% during compilation, then jump to ~100%%."
      )
    best_map = 0.0
    epoch_times = []
    t_train_start = time.time()

    # Outer epoch bar at position=0, stays visible throughout
    epoch_bar = tqdm(total=self.epochs, desc="  epochs", unit="ep",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                     dynamic_ncols=True, leave=True, position=0)

    for epoch in range(1, self.epochs+1):
      t0 = time.time()
      train_loss = self._epoch(self.train_ds, training=True, epoch_bar=epoch_bar)
      val_str = ""; map_str = ""
      if self.val_ds:
        val_loss = self._epoch(self.val_ds, training=False, epoch_bar=epoch_bar)
        val_str = f"  val={val_loss:.4f}"
        if self.eval_map:
          from .eval import evaluate
          res = evaluate(self.model, self.val_ds,
                         num_classes=self.criterion.num_classes,
                         img_size=self.img_size, batch_size=self.batch_size)
          map_str = f"  mAP={res['mAP']:.4f}"
          if res['mAP'] > best_map:
            best_map = res['mAP']
            self.model.save(os.path.join(self.save_dir, "best.safetensors"))
        elif val_loss < self.best_loss:
          self.best_loss = val_loss
          self.model.save(os.path.join(self.save_dir, "best.safetensors"))
      else:
        if train_loss < self.best_loss:
          self.best_loss = train_loss
          self.model.save(os.path.join(self.save_dir, "best.safetensors"))

      self.model.save(os.path.join(self.save_dir, "last.safetensors"))
      epoch_time = time.time() - t0
      epoch_times.append(epoch_time)
      avg_epoch = sum(epoch_times) / len(epoch_times)
      eta_total = avg_epoch * (self.epochs - epoch)

      tqdm.write(
        f"  epoch {epoch:>{len(str(self.epochs))}}/{self.epochs}"
        f"  train={train_loss:.4f}{val_str}{map_str}"
        f"  {epoch_time:.0f}s  eta {_fmt_time(eta_total)}"
      )
      epoch_bar.update(1)
      epoch_bar.set_postfix(train=f"{train_loss:.4f}", eta=_fmt_time(eta_total))

      if epoch % 10 == 0 or epoch == self.epochs:
        notify(f"RF-DETR epoch {epoch}/{self.epochs}: train={train_loss:.4f}{val_str}{map_str}")

    epoch_bar.close()
    total_time = time.time() - t_train_start
    summary = f"best_map={best_map:.4f}" if self.eval_map else f"best_loss={self.best_loss:.4f}"
    notify(f"RF-DETR training done. {summary}  total={_fmt_time(total_time)}")
    return best_map if self.eval_map else self.best_loss
