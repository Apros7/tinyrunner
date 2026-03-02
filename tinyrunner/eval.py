"""Inference, NMS, and mAP evaluation for RF-DETR."""
import numpy as np
from tinygrad import Tensor


# ── Post-processing ───────────────────────────────────────────────────────────

def nms(boxes_xyxy, scores, iou_threshold=0.5):
  """Greedy NMS. boxes_xyxy: (N,4), scores: (N,) → kept indices (sorted by score desc)."""
  if len(boxes_xyxy) == 0:
    return np.array([], dtype=int)
  order = scores.argsort()[::-1]
  x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
  areas = (x2 - x1) * (y2 - y1)
  keep = []
  while order.size:
    i = order[0]; keep.append(i)
    ix1 = np.maximum(x1[i], x1[order[1:]])
    iy1 = np.maximum(y1[i], y1[order[1:]])
    ix2 = np.minimum(x2[i], x2[order[1:]])
    iy2 = np.minimum(y2[i], y2[order[1:]])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
    order = order[1:][iou <= iou_threshold]
  return np.array(keep, dtype=int)


def _cxcywh_to_xyxy(boxes, img_h, img_w):
  """(N,4) cxcywh [0,1] → (N,4) xyxy in pixel coords."""
  cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  return np.stack([
    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
  ], axis=-1)


def postprocess(boxes_np, logits_np, img_h, img_w,
                conf_threshold=0.01, nms_iou=0.5):
  """
  Convert raw model outputs for one image to detections.
  boxes_np:  (Q, 4) cxcywh [0,1]
  logits_np: (Q, C) raw logits
  Returns: boxes_xyxy (K,4), scores (K,), labels (K,) — all numpy
  """
  probs = 1 / (1 + np.exp(-logits_np))         # (Q, C) sigmoid
  scores_all = probs.max(axis=1)                # (Q,) best class score
  labels_all = probs.argmax(axis=1)             # (Q,) best class label

  mask = scores_all >= conf_threshold
  if not mask.any():
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

  boxes_f = boxes_np[mask]
  scores_f = scores_all[mask]
  labels_f = labels_all[mask]
  boxes_xyxy = _cxcywh_to_xyxy(boxes_f, img_h, img_w)

  # Per-class NMS
  kept_boxes, kept_scores, kept_labels = [], [], []
  for cls in np.unique(labels_f):
    idx = np.where(labels_f == cls)[0]
    keep = nms(boxes_xyxy[idx], scores_f[idx], iou_threshold=nms_iou)
    kept_boxes.append(boxes_xyxy[idx[keep]])
    kept_scores.append(scores_f[idx[keep]])
    kept_labels.append(np.full(len(keep), cls, dtype=int))

  if not kept_boxes:
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
  return (np.concatenate(kept_boxes), np.concatenate(kept_scores),
          np.concatenate(kept_labels))


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, dataset, img_size=640, conf_threshold=0.01, nms_iou=0.5, batch_size=1):
  """
  Run inference over a dataset.
  Returns list of (boxes_xyxy, scores, labels) per image, all numpy arrays.
  """
  from .data import make_loader
  loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
  results = []
  with Tensor.train(False):
    for imgs, _ in loader:
      boxes_t, logits_t = model(imgs)
      boxes_np = boxes_t.numpy()    # (B, Q, 4)
      logits_np = logits_t.numpy()  # (B, Q, C)
      for i in range(boxes_np.shape[0]):
        b, s, l = postprocess(boxes_np[i], logits_np[i],
                              img_h=img_size, img_w=img_size,
                              conf_threshold=conf_threshold, nms_iou=nms_iou)
        results.append((b, s, l))
  return results


# ── AP / mAP ──────────────────────────────────────────────────────────────────

def _iou_matrix(pred_xyxy, gt_xyxy):
  """(N,4) vs (M,4) xyxy → (N,M) IoU matrix."""
  if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
    return np.zeros((len(pred_xyxy), len(gt_xyxy)))
  ix1 = np.maximum(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])
  iy1 = np.maximum(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])
  ix2 = np.minimum(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
  iy2 = np.minimum(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])
  inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
  a1 = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
  a2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
  return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


def compute_ap(tp_arr, fp_arr, n_gt, interp_points=11):
  """
  Compute AP from per-detection TP/FP arrays (sorted by score desc).
  Uses N-point interpolation (standard VOC 11-point style).
  tp_arr, fp_arr: (N,) bool/int arrays
  n_gt: total ground-truth count for this class
  """
  if n_gt == 0:
    return 0.0
  tp_cum = np.cumsum(tp_arr)
  fp_cum = np.cumsum(fp_arr)
  recalls = tp_cum / n_gt
  precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
  # Interpolated AP: at each recall threshold, take max precision at or above that recall
  ap = 0.0
  for t in np.linspace(0, 1, interp_points):
    p = precisions[recalls >= t]
    ap += (float(p.max()) if len(p) else 0.0)
  return ap / interp_points


def evaluate(model, dataset, num_classes, img_size=640,
             conf_threshold=0.01, nms_iou=0.5, iou_match=0.5, batch_size=1):
  """
  Compute per-class AP and mAP@0.5 over a dataset.

  Returns dict:
    {
      'mAP':       float,
      'AP':        {cls_id: float},
      'n_gt':      {cls_id: int},
      'n_pred':    {cls_id: int},
    }
  """
  from .data import make_loader
  loader = make_loader(dataset, batch_size=batch_size, shuffle=False)

  # Collect all predictions and ground-truths
  all_preds = {c: [] for c in range(num_classes)}  # cls → list of (score, tp, fp)
  n_gt = {c: 0 for c in range(num_classes)}

  with Tensor.train(False):
    for imgs, targets in loader:
      boxes_t, logits_t = model(imgs)
      boxes_np = boxes_t.numpy()
      logits_np = logits_t.numpy()

      for i in range(boxes_np.shape[0]):
        pb, ps, pl = postprocess(boxes_np[i], logits_np[i],
                                 img_h=img_size, img_w=img_size,
                                 conf_threshold=conf_threshold, nms_iou=nms_iou)
        tgt = targets[i]
        tb_raw = tgt["boxes"]   # (M,4) cxcywh [0,1]
        tl = tgt["labels"]      # (M,) int

        # Convert GT to xyxy pixels
        if len(tb_raw):
          tb = _cxcywh_to_xyxy(tb_raw, img_size, img_size)
        else:
          tb = np.zeros((0, 4))

        # Count GT per class
        for c in tl:
          n_gt[int(c)] += 1

        if len(pb) == 0:
          continue

        # Match predictions to GTs for each class
        gt_matched = np.zeros(len(tb), dtype=bool)
        for j in range(len(pb)):
          c = int(pl[j])
          gt_idx_c = np.where(tl == c)[0]
          tp, fp = 0, 0
          if len(gt_idx_c) and len(tb):
            iou_row = _iou_matrix(pb[j:j+1], tb[gt_idx_c])  # (1, M_c)
            best = iou_row[0].argmax()
            if iou_row[0, best] >= iou_match and not gt_matched[gt_idx_c[best]]:
              tp = 1
              gt_matched[gt_idx_c[best]] = True
            else:
              fp = 1
          else:
            fp = 1
          all_preds[c].append((ps[j], tp, fp))

  # Compute AP per class
  ap_per_class = {}
  for c in range(num_classes):
    preds = all_preds[c]
    if not preds:
      ap_per_class[c] = 0.0
      continue
    preds_sorted = sorted(preds, key=lambda x: -x[0])
    tp_arr = np.array([p[1] for p in preds_sorted])
    fp_arr = np.array([p[2] for p in preds_sorted])
    ap_per_class[c] = compute_ap(tp_arr, fp_arr, n_gt[c])

  classes_with_gt = [c for c in range(num_classes) if n_gt[c] > 0]
  mAP = np.mean([ap_per_class[c] for c in classes_with_gt]) if classes_with_gt else 0.0

  return {
    "mAP": float(mAP),
    "AP":  ap_per_class,
    "n_gt":  n_gt,
    "n_pred": {c: len(all_preds[c]) for c in range(num_classes)},
  }
