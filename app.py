from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
import onnxruntime as ort
from PIL import Image
import numpy as np, json, io, base64

app = FastAPI(title="Image Auto-Tagger", version="1.1")

# ---- model + labels (unchanged) ----
session = ort.InferenceSession("models/mobilenetv2-7.onnx", providers=["CPUExecutionProvider"])
LABELS = json.load(open("models/imagenet-simple-labels.json", "r", encoding="utf-8"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32)/255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

# ---- existing minimal endpoints ----
@app.get("/", response_class=HTMLResponse)
def home():
    return '<h3>Image Auto-Tagger</h3><p><a href="/upload">Upload UI</a> • <a href="/docs">Swagger</a></p>'

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    x = preprocess(img)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0]
    p = softmax(logits)[0]
    idx = np.argsort(-p)[:5]
    return {"predictions":[{"rank":i+1,"label":LABELS[j],"prob":float(p[j])} for i,j in enumerate(idx)]}

# ---------- New HTML template ----------
UPLOAD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Image Auto-Tagger</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;}
    .card{border:1px solid #e5e7eb;border-radius:16px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.04);}
    .row{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;}
    img{max-width:260px;height:auto;border-radius:12px;border:1px solid #eee;}
    table{border-collapse:collapse;width:100%;margin-top:10px;}
    th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;}
    .prob{font-variant-numeric:tabular-nums;}
    .btn{display:inline-block;background:#111827;color:white;border:none;padding:10px 14px;border-radius:10px;cursor:pointer}
    .muted{color:#6b7280}
  </style>
</head>
<body>
  <h2>Image Auto-Tagger</h2>
  <p class="muted">Upload a JPG/PNG. The app returns the top-5 ImageNet tags.</p>

  <div class="card">
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required />
      <button class="btn" type="submit">Analyze</button>
    </form>
  </div>

  <!--RESULT-->

  <p class="muted" style="margin-top:24px">Need APIs? Try <a href="/docs">/docs</a>. Health: <a href="/healthz">/healthz</a>.</p>
</body>
</html>
"""


def render_result(img_b64: str, preds: list):
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['label']}</td>"
        f"<td class='prob'>{r['prob']:.4f}</td></tr>"
        for r in preds
    )
    html = f"""
    <div class="card" style="margin-top:16px">
      <div class="row">
        <img src="data:image/png;base64,{img_b64}" alt="preview"/>
        <div style="flex:1;min-width:240px">
          <h3 style="margin:0 0 8px 0">Predictions</h3>
          <table>
            <thead><tr><th>#</th><th>Label</th><th>Probability</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    </div>
    """
    return html

# ---------- Updated routes ----------
@app.get("/upload", response_class=HTMLResponse)
def upload_get():
    # Replace the placeholder comment with nothing (empty result)
    return UPLOAD_HTML.replace("<!--RESULT-->", "")

@app.post("/upload", response_class=HTMLResponse)
async def upload_post(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        # Show an error message inside the page
        return UPLOAD_HTML.replace("<!--RESULT-->", "<p style='color:#b91c1c'>Invalid image file.</p>")

    # Run the model
    x = preprocess(img)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0]
    p = softmax(logits)[0]
    idx = np.argsort(-p)[:5]
    preds = [{"rank": i+1, "label": LABELS[j], "prob": float(p[j])} for i, j in enumerate(idx)]

    # Make small preview
    preview = img.copy()
    preview.thumbnail((512, 512))
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    html_result = render_result(img_b64, preds)
    return UPLOAD_HTML.replace("<!--RESULT-->", html_result)