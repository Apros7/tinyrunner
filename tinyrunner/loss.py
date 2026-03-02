"""Losses and Hungarian matching for RF-DETR."""
import numpy as np
from tinygrad import Tensor

# ── Box utilities ─────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(b):
  """(cx,cy,w,h) → (x1,y1,x2,y2), b is numpy array."""
  cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
  return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=-1)

def box_iou_np(b1, b2):
  """IoU between two sets of boxes in xyxy format. b1:(N,4), b2:(M,4) → (N,M)."""
  area1 = (b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1])
  area2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
  inter_x1 = np.maximum(b1[:,None,0], b2[None,:,0])
  inter_y1 = np.maximum(b1[:,None,1], b2[None,:,1])
  inter_x2 = np.minimum(b1[:,None,2], b2[None,:,2])
  inter_y2 = np.minimum(b1[:,None,3], b2[None,:,3])
  inter = np.maximum(0, inter_x2-inter_x1) * np.maximum(0, inter_y2-inter_y1)
  union = area1[:,None] + area2[None,:] - inter
  return inter / (union + 1e-6)

def giou_np(b1, b2):
  """Generalised IoU for matched pairs. b1,b2: (N,4) xyxy → (N,)"""
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

# ── Hungarian Matching ────────────────────────────────────────────────────────

def _hungarian(cost):
  """Solve linear assignment on (n,m) cost matrix. Returns (row_ind, col_ind)."""
  n, m = cost.shape
  INF = cost.max() * 2 + 1 if cost.size > 0 else 1.0
  sz = max(n, m)
  C = np.full((sz, sz), INF)
  C[:n, :m] = cost
  u = np.zeros(sz+1); v = np.zeros(sz+1)
  p = np.zeros(sz+1, dtype=int)
  way = np.zeros(sz+1, dtype=int)
  for i in range(1, sz+1):
    p[0] = i; j0 = 0
    minVal = np.full(sz+1, INF)
    used = np.zeros(sz+1, dtype=bool)
    while True:
      used[j0] = True
      i0, delta, j1 = p[j0], INF, -1
      for j in range(1, sz+1):
        if not used[j]:
          cur = C[i0-1, j-1] - u[i0] - v[j]
          if cur < minVal[j]:
            minVal[j] = cur
            way[j] = j0
          if minVal[j] < delta:
            delta = minVal[j]; j1 = j
      for j in range(sz+1):
        if used[j]: u[p[j]] += delta; v[j] -= delta
        else: minVal[j] -= delta
      j0 = j1
      if p[j0] == 0: break
    while j0:
      p[j0] = p[way[j0]]; j0 = way[j0]
  rows, cols = [], []
  for j in range(1, sz+1):
    if p[j] != 0 and p[j]-1 < n and j-1 < m:
      rows.append(p[j]-1); cols.append(j-1)
  return np.array(rows), np.array(cols)

def match(pred_boxes, pred_logits, tgt_boxes, tgt_labels, cls_w=2.0, l1_w=5.0, giou_w=2.0):
  """Hungarian matching for one image.
  pred_boxes: (N,4) cxcywh in [0,1] numpy
  pred_logits: (N,C) numpy
  tgt_boxes: (M,4) cxcywh in [0,1] numpy
  tgt_labels: (M,) int numpy
  Returns: (src_idx, tgt_idx) — matched prediction and target indices.
  """
  N, M = len(pred_boxes), len(tgt_boxes)
  if M == 0: return np.array([], int), np.array([], int)
  probs = 1 / (1 + np.exp(-pred_logits))
  cls_cost = -probs[:, tgt_labels]
  l1_cost = np.abs(pred_boxes[:, None] - tgt_boxes[None]).sum(-1)
  pb_xyxy = box_cxcywh_to_xyxy(pred_boxes)
  tb_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
  giou_cost = -box_iou_np(pb_xyxy, tb_xyxy)
  cost = cls_w*cls_cost + l1_w*l1_cost + giou_w*giou_cost
  return _hungarian(cost)

# ── Criterion ─────────────────────────────────────────────────────────────────

class SetCriterion:
  def __init__(self, num_classes, cls_w=2.0, l1_w=5.0, giou_w=2.0, focal_alpha=0.25, focal_gamma=2.0):
    self.num_classes = num_classes
    self.cls_w, self.l1_w, self.giou_w = cls_w, l1_w, giou_w
    self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma

  def _focal_loss(self, logits, targets):
    """Binary focal loss. logits: (N,C), targets: (N,C) float."""
    p = logits.sigmoid()
    ce = logits.binary_crossentropy_logits(targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
    return (alpha_t * (1 - p_t)**self.focal_gamma * ce).mean()

  def compute_matches(self, pb, pl, targets):
    """Compute Hungarian matching given numpy predictions and targets.
    pb: (B, N, 4) numpy — predicted boxes (cxcywh)
    pl: (B, N, C) numpy — predicted logits
    Returns: list of (src_idx, tgt_idx) per image
    """
    B = pb.shape[0]
    matches = []
    for i in range(B):
      tgt = targets[i]
      tb, tl = tgt["boxes"], tgt["labels"]
      if len(tb) == 0:
        matches.append((np.array([], int), np.array([], int)))
      else:
        src_idx, tgt_idx = match(pb[i], pl[i], tb, tl, self.cls_w, self.l1_w, self.giou_w)
        matches.append((src_idx, tgt_idx))
    return matches

  def __call__(self, pred_boxes, pred_logits, targets, matches=None, match_pb=None):
    """
    pred_boxes:  (B, N, 4) tinygrad Tensor — must NOT be realized before calling
    pred_logits: (B, N, C) tinygrad Tensor — must NOT be realized before calling
    targets: list of dicts {'boxes': (M,4) numpy, 'labels': (M,) numpy int}
    matches: pre-computed list of (src_idx, tgt_idx); if None, computed internally
             (causes realization of pred_boxes/pred_logits — OK for validation only)
    match_pb: (B, N, 4) numpy array of predictions used for GIoU cost; required if matches given
    """
    B = pred_boxes.shape[0]

    if matches is None:
      # Validation mode: realize preds to compute matching (no grad needed)
      pb = pred_boxes.numpy(); pl = pred_logits.numpy()
      matches = self.compute_matches(pb, pl, targets)
      match_pb = pb

    cls_loss = Tensor(0.0); box_loss = Tensor(0.0); giou_loss = Tensor(0.0)
    num_boxes = 0

    for i in range(B):
      src_idx, tgt_idx = matches[i]
      tgt = targets[i]
      tb, tl = tgt["boxes"], tgt["labels"]
      if len(src_idx) == 0: continue
      num_boxes += len(src_idx)

      # classification: focal loss over all queries
      tgt_cls = np.zeros((pred_boxes.shape[1], self.num_classes), dtype=np.float32)
      tgt_cls[src_idx, tl[tgt_idx]] = 1.0
      cls_loss = cls_loss + self._focal_loss(pred_logits[i], Tensor(tgt_cls))

      # box regression: L1 on matched pairs
      src_list = src_idx.tolist()
      matched_pb = pred_boxes[i][src_list]
      matched_tb = Tensor(tb[tgt_idx].astype(np.float32))
      box_loss = box_loss + (matched_pb - matched_tb).abs().sum() / len(src_idx)

      # GIoU on matched pairs (numpy-based, no grad through pred_boxes)
      pb_xy = box_cxcywh_to_xyxy(match_pb[i][src_idx])
      tb_xy = box_cxcywh_to_xyxy(tb[tgt_idx])
      g = giou_np(pb_xy, tb_xy)
      giou_loss = giou_loss + Tensor(1.0 - g).mean()

    scale = max(1, num_boxes)
    total = (self.cls_w * cls_loss + self.l1_w * box_loss + self.giou_w * giou_loss) / scale
    # Return sub-losses as lazy Tensors — caller must realize AFTER backward
    return total, {"cls": cls_loss / scale, "box": box_loss / scale, "giou": giou_loss / scale}
