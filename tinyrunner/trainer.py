"""Trainer for RF-DETR."""
import time, os, math, sys
from tinygrad import Tensor, nn
from tinygrad.nn import state as nn_state
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

  def _epoch(self, dataset, training):
    from .data import make_loader
    loader = make_loader(dataset, self.batch_size, shuffle=training)
    total_loss = 0.0; steps = 0; n = len(dataset) // self.batch_size
    t_epoch = time.time(); t_step = time.time()
    for imgs, targets in loader:
      if steps >= n: break
      if training:
        self._step()
        # Pass 1: eval mode to get numpy preds for matching
        with Tensor.train(False):
          boxes_m, logits_m = self.model(imgs)
        pb = boxes_m.numpy(); pl = logits_m.numpy()
        matches = self.criterion.compute_matches(pb, pl, targets)
        # Pass 2: fresh forward in train mode → backward → step
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
      now = time.time()
      step_time = now - t_step; t_step = now
      elapsed = now - t_epoch
      avg_step = elapsed / steps
      eta_epoch = avg_step * (n - steps)
      imgs_per_sec = self.batch_size / max(step_time, 1e-6)
      tag = "train" if training else "val"
      print(
        f"  {tag} {steps:>{len(str(n))}}/{n}"
        f"  loss={total_loss/steps:.4f}"
        f"  cls={sub_vals['cls']:.3f} box={sub_vals['box']:.3f} giou={sub_vals['giou']:.3f}"
        f"  {imgs_per_sec:.1f}img/s"
        f"  eta {_fmt_time(eta_epoch)}",
        flush=True,
      )
    return total_loss / max(1, steps)

  def train(self):
    notify("RF-DETR training started")
    print(f"Training RF-DETR for {self.epochs} epochs, batch={self.batch_size}, lr={self.lr}", flush=True)
    best_map = 0.0
    epoch_times = []
    t_train_start = time.time()
    for epoch in range(1, self.epochs+1):
      t0 = time.time()
      print(f"\nEpoch {epoch}/{self.epochs}", flush=True)
      train_loss = self._epoch(self.train_ds, training=True)
      val_str = ""; map_str = ""
      if self.val_ds:
        val_loss = self._epoch(self.val_ds, training=False)
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
      print(
        f"  --> epoch {epoch}/{self.epochs}  train={train_loss:.4f}{val_str}{map_str}"
        f"  {epoch_time:.0f}s/epoch  eta {_fmt_time(eta_total)}",
        flush=True,
      )

      if epoch % 10 == 0 or epoch == self.epochs:
        notify(f"RF-DETR epoch {epoch}/{self.epochs}: train={train_loss:.4f}{val_str}{map_str}")

    total_time = time.time() - t_train_start
    summary = f"best_map={best_map:.4f}" if self.eval_map else f"best_loss={self.best_loss:.4f}"
    notify(f"RF-DETR training done. {summary}  total={_fmt_time(total_time)}")
    return best_map if self.eval_map else self.best_loss
