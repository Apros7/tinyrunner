"""Trainer for RF-DETR."""
import time, os, math
from tinygrad import Tensor, nn
from tinygrad.nn import state as nn_state
from .notify import notify

class Trainer:
  def __init__(self, model, criterion, train_ds, val_ds=None,
               lr=1e-4, batch_size=4, epochs=50, img_size=640,
               save_dir="runs", pretrained_backbone=True, warmup_epochs=2):
    self.model = model; self.criterion = criterion
    self.train_ds = train_ds; self.val_ds = val_ds
    self.epochs = epochs; self.batch_size = batch_size
    self.save_dir = save_dir; self.lr = lr
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
    for imgs, targets in loader:
      if steps >= n: break
      if training:
        self._step()
        # Pass 1: eval mode to get numpy preds for matching (realizes intermediates — OK, no grad needed)
        with Tensor.train(False):
          boxes_m, logits_m = self.model(imgs)
        pb = boxes_m.numpy(); pl = logits_m.numpy()
        matches = self.criterion.compute_matches(pb, pl, targets)
        # Pass 2: fresh forward in train mode — no .numpy() on outputs before backward
        with Tensor.train():
          boxes, logits = self.model(imgs)
          loss, sub = self.criterion(boxes, logits, targets, matches=matches, match_pb=pb)
          loss.backward()
          self._update()
        # Realize after backward+step to avoid breaking computation graph
        loss_val = float(loss.numpy())
        sub_vals = {k: float(v.numpy()) for k, v in sub.items()}
      else:
        with Tensor.train(False):
          boxes, logits = self.model(imgs)
          loss, sub = self.criterion(boxes, logits, targets)
        loss_val = float(loss.numpy())
        sub_vals = {k: float(v.numpy()) for k, v in sub.items()}
      total_loss += loss_val; steps += 1
      if steps % 20 == 0:
        print(f"  step {steps}/{n}  loss={total_loss/steps:.4f}  "
              f"cls={sub_vals['cls']:.4f} box={sub_vals['box']:.4f} giou={sub_vals['giou']:.4f}", flush=True)
    return total_loss / max(1, steps)

  def train(self):
    notify("🚀 RF-DETR training started")
    print(f"Training RF-DETR for {self.epochs} epochs, batch={self.batch_size}, lr={self.lr}")
    for epoch in range(1, self.epochs+1):
      t0 = time.time()
      train_loss = self._epoch(self.train_ds, training=True)
      val_str = ""
      if self.val_ds:
        val_loss = self._epoch(self.val_ds, training=False)
        val_str = f"  val={val_loss:.4f}"
        if val_loss < self.best_loss:
          self.best_loss = val_loss
          self.model.save(os.path.join(self.save_dir, "best.safetensors"))

      self.model.save(os.path.join(self.save_dir, "last.safetensors"))
      print(f"Epoch {epoch}/{self.epochs}  train={train_loss:.4f}{val_str}  {time.time()-t0:.1f}s", flush=True)

      if epoch % 10 == 0 or epoch == self.epochs:
        notify(f"RF-DETR epoch {epoch}/{self.epochs}: train={train_loss:.4f}{val_str}")

    notify(f"✅ RF-DETR training done. Best loss: {self.best_loss:.4f}")
    return self.best_loss
