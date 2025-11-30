from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os, io, time
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import timm

# ------------------------
# Config
# ------------------------
MODEL_PATH = "saved_models/best_vit_model.pth"  # change if needed
MODEL_NAME = 'vit_tiny_patch16_224'
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Developer-provided diagram path (used as a static illustration in the UI)
DIAGRAM_PATH = "/mnt/data/A_diagram_in_this_digital_illustration_depicts_the.png"

# ------------------------
# Load model
# ------------------------
print("Loading model...", MODEL_NAME)
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
        print("Loaded weights from", MODEL_PATH)
    except Exception as e:
        print("Warning: could not load state_dict directly:", e)
else:
    print("Warning: checkpoint not found at:", MODEL_PATH)
model.to(DEVICE).eval()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

def load_image_for_model(path):
    img = Image.open(path).convert('RGB')
    orig = np.array(img)
    t = transform(img).unsqueeze(0).to(DEVICE)
    return t, orig

# ------------------------
# Attention rollout function for timm ViT
# ------------------------
@torch.no_grad()
def attention_rollout_map(model, x, discard_ratio=0.85):
    attn_blocks = []

    # forward hook that recomputes attention from qkv
    def hook_attention(module, inputs, output):
        tokens = inputs[0]  # [B, N, C]
        B, N, C = tokens.shape
        # qkv projection
        qkv = module.qkv(tokens)  # [B, N, 3*C]
        num_heads = module.num_heads
        head_dim = C // num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
        attn = attn.softmax(dim=-1)  # [B, heads, N, N]
        attn_blocks.append(attn.detach().cpu())

    hooks = [blk.attn.register_forward_hook(hook_attention) for blk in model.blocks]
    _ = model(x)  # forward pass triggers hooks
    for h in hooks:
        h.remove()

    if len(attn_blocks) == 0:
        raise RuntimeError("No attention maps captured. Check model structure or hooks.")

    # Combine layers (rollout)
    attns = [a[0].mean(dim=0).numpy() for a in attn_blocks]  # each [N, N]
    result = np.eye(attns[0].shape[-1], dtype=np.float32)
    for attn in attns:
        A = attn.copy()
        flat = A.reshape(-1)
        k = max(1, int(flat.size * (1 - discard_ratio)))
        thresh = np.partition(flat, -k)[-k]
        A[A < thresh] = 0.0
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-6)
        result = result @ A

    mask = result[0, 1:]
    side = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(side, side)
    mask = cv2.resize(mask, (224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask

# ------------------------
# Overlay helper
# ------------------------

def overlay_attention(orig_img_rgb, attn_map_2d, alpha=0.45):
    H, W = orig_img_rgb.shape[:2]
    attn_resized = cv2.resize(attn_map_2d, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR), 1-alpha, heatmap, alpha, 0)
    return overlay, attn_resized

# ------------------------
# Flask app + UI (single-file template with glassmorphism + animation)
# ------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ViT Attention Visualizer</title>
  <style>
    :root{--bg:#0f1724;--card:rgba(255,255,255,0.06);--accent:#7c3aed}
    body{margin:0;font-family:Inter,system-ui,Segoe UI,Roboto,'Helvetica Neue',Arial;background:linear-gradient(180deg,#0b1220 0%, #192231 100%);color:#e6eef8}
    .container{max-width:1100px;margin:32px auto;padding:24px}
    .header{display:flex;align-items:center;gap:16px}
    .brand{font-weight:700;font-size:20px}
    .sub{color:#9fb0c8;font-size:13px}
    .uploader{margin-top:20px;background:var(--card);backdrop-filter:blur(8px);padding:18px;border-radius:14px;box-shadow:0 6px 30px rgba(2,6,23,0.6)}
    .drop{border:2px dashed rgba(255,255,255,0.06);padding:28px;text-align:center;border-radius:10px}
    .btn{background:linear-gradient(90deg,var(--accent),#5b21b6);color:white;padding:10px 16px;border-radius:10px;border:none;cursor:pointer}
    .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:18px}
    .card{background:rgba(255,255,255,0.03);padding:14px;border-radius:12px;min-height:300px;display:flex;align-items:center;justify-content:center}
    img.resp{max-width:100%;max-height:420px;border-radius:10px;box-shadow:0 8px 30px rgba(2,6,23,0.6)}
    .spinner{width:48px;height:48px;border-radius:50%;border:4px solid rgba(255,255,255,0.08);border-left-color:var(--accent);animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    footer{margin-top:18px;text-align:center;color:#9fb0c8}
    .small{font-size:13px;color:#9fb0c8}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div style="width:56px;height:56px;border-radius:12px;background:linear-gradient(180deg,#5b21b6,#7c3aed);display:flex;align-items:center;justify-content:center;font-weight:700">ViT</div>
      <div>
        <div class="brand">ViT Attention Visualizer</div>
        <div class="sub">Upload a blood-cell image, get attention heatmap + overlay (glass UI)</div>
      </div>
    </div>

    <div class="uploader">
      <form id="upload-form" method="POST" action="/predict" enctype="multipart/form-data">
        <div class="drop">
          <input type="file" name="file" accept="image/*" required />
          <div style="height:8px"></div>
          <button class="btn" type="submit">Generate Visualization</button>
        </div>
      </form>
      <div style="margin-top:12px" class="small">Tip: Use a cropped cell image for best results. A diagram below explains attention rollout:</div>
      <div style="margin-top:12px"><img src="{{ diagram_url }}" alt="diagram" style="max-width:420px;border-radius:8px;box-shadow:0 12px 40px rgba(0,0,0,0.6)"></div>
    </div>

    {% if results %}
    <div class="grid">
      <div class="card">
        <div style="text-align:center">
          <div class="small">Original</div>
          <img class="resp" src="{{ results.original }}">
        </div>
      </div>
      <div class="card">
        <div style="text-align:center">
          <div class="small">Attention Map</div>
          <img class="resp" src="{{ results.heatmap }}">
        </div>
      </div>
      <div class="card">
        <div style="text-align:center">
          <div class="small">Overlay</div>
          <img class="resp" src="{{ results.overlay }}">
        </div>
      </div>
    </div>
    {% endif %}

    <footer>
      <div class="small">Model: {{ model_name }} • Device: {{ device }}</div>
    </footer>
  </div>

  <script>
    const form = document.getElementById('upload-form');
    form.addEventListener('submit', () => {
      // simple UX: show spinner while processing
      const btn = form.querySelector('button');
      btn.disabled = true; btn.textContent = 'Processing...';
    });
  </script>
</body>
</html>
"""

# ------------------------
# Flask routes
# ------------------------
@app.route('/')
def index():
    return render_template_string(TEMPLATE, results=None, diagram_url=DIAGRAM_PATH, model_name=MODEL_NAME, device=str(DEVICE))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    timestamp = int(time.time()*1000)
    fname = f"upload_{timestamp}.png"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)

    # run model & attention
    x, orig = load_image_for_model(fpath)
    try:
        attn_map = attention_rollout_map(model, x, discard_ratio=0.88)
    except Exception as e:
        return f"Error generating attention map: {e}", 500

    overlay_bgr, _ = overlay_attention(orig, attn_map)

    # save outputs
    heat_name = f"heat_{timestamp}.png"
    over_name = f"overlay_{timestamp}.png"
    heat_path = os.path.join(app.config['RESULT_FOLDER'], heat_name)
    over_path = os.path.join(app.config['RESULT_FOLDER'], over_name)

    cv2.imwrite(heat_path, cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET))
    cv2.imwrite(over_path, overlay_bgr)

    results = {
        'original': '/' + fpath,
        'heatmap': '/' + heat_path,
        'overlay': '/' + over_path
    }
    return render_template_string(TEMPLATE, results=results, diagram_url=DIAGRAM_PATH, model_name=MODEL_NAME, device=str(DEVICE))

# Serve static files by default from /static. Also allow direct serving of uploaded files outside static
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print('Starting Flask app on http://127.0.0.1:5000')
    app.run(debug=True)
