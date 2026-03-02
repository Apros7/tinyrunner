"""RF-DETR: Real-time Feature Detection Transformer in tinygrad."""
import math
from tinygrad import Tensor, nn
from tinygrad.nn import state as nn_state

# ── Backbone ──────────────────────────────────────────────────────────────────

class Bottleneck:
  expansion = 4
  def __init__(self, in_c, c, stride=1):
    w = c
    self.conv1 = nn.Conv2d(in_c, w, 1, bias=False);       self.bn1 = nn.BatchNorm(w)
    self.conv2 = nn.Conv2d(w, w, 3, stride=stride, padding=1, bias=False); self.bn2 = nn.BatchNorm(w)
    self.conv3 = nn.Conv2d(w, c*4, 1, bias=False);        self.bn3 = nn.BatchNorm(c*4)
    self.downsample = [] if stride == 1 and in_c == c*4 else \
      [nn.Conv2d(in_c, c*4, 1, stride=stride, bias=False), nn.BatchNorm(c*4)]
  def __call__(self, x):
    return (self.bn3(self.conv3(self.bn2(self.conv2(self.bn1(self.conv1(x)).relu())).relu()))
            + x.sequential(self.downsample)).relu()

def _make_layer(in_c, c, n, stride=1):
  return [Bottleneck(in_c, c, stride)] + [Bottleneck(c*4, c) for _ in range(1, n)]

class ResNet50:
  """ResNet50 backbone returning (C3, C4, C5) feature maps."""
  def __init__(self):
    self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False); self.bn1 = nn.BatchNorm(64)
    self.layer1 = _make_layer(64, 64, 3)
    self.layer2 = _make_layer(256, 128, 4, stride=2)
    self.layer3 = _make_layer(512, 256, 6, stride=2)
    self.layer4 = _make_layer(1024, 512, 3, stride=2)

  def __call__(self, x):
    x = self.bn1(self.conv1(x)).relu().pad([1,1,1,1]).max_pool2d(3, 2)
    x = x.sequential(self.layer1)
    c3 = x.sequential(self.layer2)   # stride 8,  512ch
    c4 = c3.sequential(self.layer3)  # stride 16, 1024ch
    c5 = c4.sequential(self.layer4)  # stride 32, 2048ch
    return c3, c4, c5

  def load_pretrained(self):
    from tinygrad.helpers import fetch
    from tinygrad.nn.state import torch_load, get_state_dict
    url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    sd = torch_load(fetch(url))
    cur = get_state_dict(self)
    for k, v in sd.items():
      if k in cur and cur[k].shape == v.shape:
        cur[k].assign(v.to(cur[k].device))
    print("Loaded pretrained ResNet50 backbone")

# ── Attention ─────────────────────────────────────────────────────────────────

class MultiheadAttention:
  def __init__(self, d, heads):
    self.h, self.dh = heads, d // heads
    self.q_proj = nn.Linear(d, d);  self.k_proj = nn.Linear(d, d)
    self.v_proj = nn.Linear(d, d);  self.out_proj = nn.Linear(d, d)

  def __call__(self, q, k=None, v=None, mask=None):
    """Self-attention if k=None, cross-attention otherwise."""
    if k is None: k = q
    if v is None: v = k
    B, Nq, D = q.shape
    Nk = k.shape[1]
    def proj_heads(x, proj, N):
      return proj(x).reshape(B, N, self.h, self.dh).transpose(1, 2)  # (B,H,N,Dh)
    q_ = proj_heads(q, self.q_proj, Nq)
    k_ = proj_heads(k, self.k_proj, Nk)
    v_ = proj_heads(v, self.v_proj, Nk)
    attn = q_.scaled_dot_product_attention(k_, v_, attn_mask=mask)
    return self.out_proj(attn.transpose(1, 2).reshape(B, Nq, D))

# ── Transformer Building Block ────────────────────────────────────────────────

class TransformerLayer:
  def __init__(self, d, heads, ffn_d, dropout=0.0):
    self.self_attn = MultiheadAttention(d, heads)
    self.cross_attn = MultiheadAttention(d, heads)
    self.ff1 = nn.Linear(d, ffn_d); self.ff2 = nn.Linear(ffn_d, d)
    self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
    self.dropout = dropout

  def __call__(self, q, mem, mem_mask=None):
    q = self.n1(q + self.self_attn(q).dropout(self.dropout))
    q = self.n2(q + self.cross_attn(q, mem, mem).dropout(self.dropout))
    q = self.n3(q + self.ff2(self.ff1(q).gelu()).dropout(self.dropout))
    return q

# ── Encoder ───────────────────────────────────────────────────────────────────

def sincos2d(h, w, d):
  """2D sine-cosine positional encoding (numpy, computed once). Returns (h*w, d)."""
  import numpy as np
  assert d % 4 == 0
  freq = np.array([10000 ** (2*i/d) for i in range(d//4)], dtype=np.float32)
  y = np.arange(h, dtype=np.float32)[:, None] / freq  # (h, d//4)
  x = np.arange(w, dtype=np.float32)[:, None] / freq  # (w, d//4)
  # row encodes y-position, col encodes x-position; interleave for each grid cell
  row = np.concatenate([np.sin(y), np.cos(y)], axis=-1)  # (h, d//2)
  col = np.concatenate([np.sin(x), np.cos(x)], axis=-1)  # (w, d//2)
  # broadcast: each cell (i,j) gets [row[i], col[j]]
  pe = np.concatenate([
    np.tile(row[:, None, :], (1, w, 1)),  # (h, w, d//2)
    np.tile(col[None, :, :], (h, 1, 1)),  # (h, w, d//2)
  ], axis=-1).reshape(h*w, d)             # (h*w, d)
  return Tensor(pe)

class EncoderLayer:
  """Single transformer encoder layer for AIFI."""
  def __init__(self, d, heads, ffn_d):
    self.attn = MultiheadAttention(d, heads)
    self.ff1 = nn.Linear(d, ffn_d); self.ff2 = nn.Linear(ffn_d, d)
    self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)

  def __call__(self, x):
    x = self.n1(x + self.attn(x))
    return self.n2(x + self.ff2(self.ff1(x).gelu()))

def _upsample(x, h, w):
  """Nearest-neighbor upsample to (h, w)."""
  B, C, ih, iw = x.shape
  eh = math.ceil(h / ih); ew = math.ceil(w / iw)
  return x.reshape(B, C, ih, 1, iw, 1).expand(B, C, ih, eh, iw, ew).reshape(B, C, ih*eh, iw*ew)[:, :, :h, :w]

def _downsample(x, h, w):
  """Stride-2 downsample to exactly (h, w) via slice."""
  return x[:, :, ::2, ::2][:, :, :h, :w]

class HybridEncoder:
  """RT-DETR Hybrid Encoder: AIFI on C5 + CCFM multi-scale fusion."""
  def __init__(self, d=256, enc_layers=1, heads=8):
    # input projections C3(512), C4(1024), C5(2048) → d
    self.proj3 = nn.Conv2d(512, d, 1, bias=False);  self.bn3 = nn.BatchNorm(d)
    self.proj4 = nn.Conv2d(1024, d, 1, bias=False); self.bn4 = nn.BatchNorm(d)
    self.proj5 = nn.Conv2d(2048, d, 1, bias=False); self.bn5 = nn.BatchNorm(d)
    # AIFI: transformer encoder on C5
    self.aifi = [EncoderLayer(d, heads, d*4) for _ in range(enc_layers)]
    # CCFM: top-down fusion convs
    self.td_c4 = nn.Conv2d(d*2, d, 1, bias=False); self.td_bn4 = nn.BatchNorm(d)
    self.td_c3 = nn.Conv2d(d*2, d, 1, bias=False); self.td_bn3 = nn.BatchNorm(d)
    # CCFM: bottom-up fusion convs
    self.bu_c4 = nn.Conv2d(d*2, d, 1, bias=False); self.bu_bn4 = nn.BatchNorm(d)
    self.bu_c5 = nn.Conv2d(d*2, d, 1, bias=False); self.bu_bn5 = nn.BatchNorm(d)
    self.d = d

  def __call__(self, c3, c4, c5):
    # project to d channels
    p3 = self.bn3(self.proj3(c3)).relu()
    p4 = self.bn4(self.proj4(c4)).relu()
    p5 = self.bn5(self.proj5(c5)).relu()
    # AIFI on p5
    B, _, H5, W5 = p5.shape
    flat = p5.permute(0,2,3,1).reshape(B, H5*W5, self.d)
    flat = flat.sequential(self.aifi)
    p5 = flat.reshape(B, H5, W5, self.d).permute(0,3,1,2)
    # top-down
    _, _, H4, W4 = p4.shape; _, _, H3, W3 = p3.shape
    f4 = self.td_bn4(self.td_c4(Tensor.cat(p4, _upsample(p5, H4, W4), dim=1))).relu()
    f3 = self.td_bn3(self.td_c3(Tensor.cat(p3, _upsample(f4, H3, W3), dim=1))).relu()
    # bottom-up (stride-2 to match each scale's exact spatial size)
    _, _, H4, W4 = f4.shape; _, _, H5, W5 = p5.shape
    g4 = self.bu_bn4(self.bu_c4(Tensor.cat(f4, _downsample(f3, H4, W4), dim=1))).relu()
    g5 = self.bu_bn5(self.bu_c5(Tensor.cat(p5, _downsample(g4, H5, W5), dim=1))).relu()
    return f3, g4, g5  # (B,d,H3,W3), (B,d,H4,W4), (B,d,H5,W5)

# ── Decoder ───────────────────────────────────────────────────────────────────

class TransformerDecoder:
  def __init__(self, num_queries=300, d=256, heads=8, num_layers=6, ffn_d=1024):
    self.num_queries = num_queries
    self.query_embed = nn.Embedding(num_queries, d)
    self.query_pos   = nn.Embedding(num_queries, d)
    self.layers = [TransformerLayer(d, heads, ffn_d) for _ in range(num_layers)]
    self.norm = nn.LayerNorm(d)
    self.d = d

  def __call__(self, memory):
    """memory: (B, L, d) — concatenated encoder features."""
    B = memory.shape[0]
    q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
    pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
    q = q + pos
    for layer in self.layers:
      q = layer(q, memory)
    return self.norm(q)  # (B, num_queries, d)

# ── Detection Heads ───────────────────────────────────────────────────────────

class MLP:
  def __init__(self, in_d, hidden_d, out_d, n_layers):
    dims = [in_d] + [hidden_d]*(n_layers-1) + [out_d]
    self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(n_layers)]
  def __call__(self, x):
    for l in self.layers[:-1]: x = l(x).relu()
    return self.layers[-1](x)

# ── RF-DETR ───────────────────────────────────────────────────────────────────

class RFDETR:
  """RF-DETR: ResNet50 backbone + Hybrid Encoder + Transformer Decoder."""
  def __init__(self, num_classes=80, num_queries=300, d=256, enc_layers=1,
               dec_layers=6, heads=8, ffn_d=1024):
    self.backbone = ResNet50()
    self.encoder  = HybridEncoder(d, enc_layers, heads)
    self.decoder  = TransformerDecoder(num_queries, d, heads, dec_layers, ffn_d)
    self.box_head = MLP(d, d, 4, 3)        # → (cx,cy,w,h) in [0,1]
    self.cls_head = nn.Linear(d, num_classes)  # → class logits (sigmoid focal)
    self.d = d

  def __call__(self, x):
    """Forward pass. x: (B,3,H,W). Returns (pred_boxes, pred_logits)."""
    c3, c4, c5 = self.backbone(x)
    f3, g4, g5 = self.encoder(c3, c4, c5)
    # flatten and concat encoder output with positional encoding
    B = x.shape[0]
    def flatten_with_pos(feat):
      _, _, H, W = feat.shape
      pos = sincos2d(H, W, self.d).unsqueeze(0).expand(B, -1, -1)
      return feat.permute(0,2,3,1).reshape(B, H*W, self.d) + pos
    memory = Tensor.cat(flatten_with_pos(f3), flatten_with_pos(g4), flatten_with_pos(g5), dim=1)
    hs = self.decoder(memory)             # (B, num_queries, d)
    boxes  = self.box_head(hs).sigmoid()  # (B, num_queries, 4)
    logits = self.cls_head(hs)            # (B, num_queries, num_classes)
    return boxes, logits

  def load_pretrained_backbone(self):
    self.backbone.load_pretrained()

  def save(self, path):
    nn_state.safe_save(nn_state.get_state_dict(self), path)
    print(f"Saved model to {path}")

  def load(self, path):
    nn_state.load_state_dict(self, nn_state.safe_load(path))
    print(f"Loaded model from {path}")
