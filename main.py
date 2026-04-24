"""
Sentiment Analysis API — Chainguard PyTorch Demo

Serves a self-contained, single-page web UI backed by a DistilBERT
sentiment-analysis pipeline. All inference runs locally inside the
Chainguard PyTorch container — no external API calls.

Endpoints:
  GET  /         → web UI (HTML)
  POST /analyze  → {"text": "..."} → sentiment result JSON
  GET  /health   → {"status": "ok"} (used by Docker healthcheck)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_classifier: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    print(f"Loading model: {MODEL_NAME}", flush=True)
    start = time.time()
    _classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        # Run on CPU in the container; GPU will be used automatically if available
        device=-1 if not os.getenv("CUDA_VISIBLE_DEVICES") else 0,
    )
    print(f"Model ready in {time.time() - start:.1f}s", flush=True)
    yield
    _classifier = None


app = FastAPI(title="Sentiment Analysis — Chainguard AI Demo", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# API schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to classify")


class SentimentResult(BaseModel):
    label: str
    score: float
    confidence_pct: int
    emoji: str
    color: str
    latency_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": _classifier is not None}


@app.post("/analyze", response_model=SentimentResult)
def analyze(req: AnalyzeRequest):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    t0 = time.perf_counter()
    result = _classifier(req.text, truncation=True, max_length=512)[0]
    latency_ms = int((time.perf_counter() - t0) * 1000)

    label: str = result["label"]      # "POSITIVE" or "NEGATIVE"
    score: float = result["score"]    # 0.0 – 1.0

    return SentimentResult(
        label=label,
        score=score,
        confidence_pct=int(score * 100),
        emoji="😊" if label == "POSITIVE" else "😞",
        color="#22c55e" if label == "POSITIVE" else "#ef4444",
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Web UI — single-page app served directly from Python
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Chainguard AI — Sentiment Demo</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0b0b18;
      --surface: #13132a;
      --surface2: #1c1c3a;
      --border: #2a2a55;
      --accent: #7c3aed;
      --accent-light: #a78bfa;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --positive: #22c55e;
      --negative: #ef4444;
      --radius: 14px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem 4rem;
    }

    /* ── Header ─────────────────────────────────────────────────── */
    header {
      text-align: center;
      margin-bottom: 2.5rem;
      animation: fadeDown .5s ease;
    }

    .logo {
      display: inline-flex;
      align-items: center;
      gap: .6rem;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 99px;
      padding: .35rem 1rem .35rem .5rem;
      font-size: .8rem;
      color: var(--muted);
      margin-bottom: 1.2rem;
    }

    .logo img { height: 22px; }

    h1 {
      font-size: clamp(1.8rem, 4vw, 2.6rem);
      font-weight: 800;
      background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
      margin-bottom: .5rem;
    }

    .subtitle {
      color: var(--muted);
      font-size: .95rem;
    }

    .badge {
      display: inline-block;
      background: rgba(124,58,237,.15);
      border: 1px solid rgba(124,58,237,.4);
      color: var(--accent-light);
      border-radius: 99px;
      padding: .2rem .75rem;
      font-size: .75rem;
      font-weight: 600;
      margin-top: .6rem;
    }

    /* ── Main card ───────────────────────────────────────────────── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      width: 100%;
      max-width: 680px;
      box-shadow: 0 20px 60px rgba(0,0,0,.4);
      animation: fadeUp .5s ease .1s both;
    }

    label {
      display: block;
      font-size: .85rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: .5rem;
    }

    textarea {
      width: 100%;
      min-height: 120px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 1rem;
      line-height: 1.6;
      padding: .85rem 1rem;
      resize: vertical;
      transition: border-color .2s;
      outline: none;
    }

    textarea:focus { border-color: var(--accent); }
    textarea::placeholder { color: #4a4a6a; }

    .char-count {
      text-align: right;
      font-size: .75rem;
      color: var(--muted);
      margin-top: .3rem;
    }

    /* ── Presets ─────────────────────────────────────────────────── */
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: .5rem;
      margin: 1rem 0;
    }

    .preset {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--muted);
      font-size: .78rem;
      padding: .3rem .7rem;
      cursor: pointer;
      transition: all .15s;
    }

    .preset:hover {
      border-color: var(--accent-light);
      color: var(--accent-light);
    }

    /* ── Button ──────────────────────────────────────────────────── */
    button#analyzeBtn {
      width: 100%;
      margin-top: 1rem;
      padding: .9rem;
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
      border: none;
      border-radius: 10px;
      color: #fff;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: opacity .2s, transform .1s;
      letter-spacing: .02em;
    }

    button#analyzeBtn:hover { opacity: .9; }
    button#analyzeBtn:active { transform: scale(.98); }
    button#analyzeBtn:disabled { opacity: .5; cursor: not-allowed; }

    /* ── Result panel ─────────────────────────────────────────────── */
    #result {
      margin-top: 1.5rem;
      animation: fadeUp .35s ease;
    }

    .result-card {
      background: var(--surface2);
      border-radius: 12px;
      padding: 1.5rem;
      border: 1px solid var(--border);
    }

    .result-header {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1.2rem;
    }

    .emoji { font-size: 3rem; line-height: 1; }

    .label-group { flex: 1; }

    .sentiment-label {
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: .03em;
    }

    .confidence-text {
      font-size: .85rem;
      color: var(--muted);
      margin-top: .1rem;
    }

    .latency {
      font-size: .75rem;
      color: var(--muted);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 99px;
      padding: .2rem .6rem;
      white-space: nowrap;
    }

    /* Progress bar */
    .bar-track {
      background: rgba(255,255,255,.06);
      border-radius: 99px;
      height: 10px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width .6s cubic-bezier(.16,1,.3,1);
    }

    .bar-labels {
      display: flex;
      justify-content: space-between;
      font-size: .72rem;
      color: var(--muted);
      margin-top: .4rem;
    }

    /* ── Error ───────────────────────────────────────────────────── */
    .error {
      background: rgba(239,68,68,.1);
      border: 1px solid rgba(239,68,68,.3);
      color: #fca5a5;
      border-radius: 10px;
      padding: 1rem;
      font-size: .9rem;
    }

    /* ── Info strip ──────────────────────────────────────────────── */
    .info-strip {
      max-width: 680px;
      width: 100%;
      margin-top: 1.5rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: .75rem;
      animation: fadeUp .5s ease .2s both;
    }

    .info-pill {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: .75rem 1rem;
    }

    .info-pill .pill-label {
      font-size: .7rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: .2rem;
    }

    .info-pill .pill-value {
      font-size: .9rem;
      font-weight: 600;
      color: var(--accent-light);
    }

    /* ── Animations ──────────────────────────────────────────────── */
    @keyframes fadeDown {
      from { opacity: 0; transform: translateY(-16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      vertical-align: middle;
      margin-right: .4rem;
    }
  </style>
</head>
<body>

  <header>
    <div class="logo">
      <span>🔗</span>
      <span>Powered by Chainguard PyTorch &amp; DistilBERT</span>
    </div>
    <h1>Sentiment Analysis</h1>
    <p class="subtitle">Local inference — no data leaves this container</p>
    <span class="badge">SLSA Level 3 · Signed · Zero CVE base image</span>
  </header>

  <div class="card">
    <label for="inputText">Text to analyse</label>
    <textarea
      id="inputText"
      placeholder="Type something — positive or negative — and see the model classify it in real time…"
      maxlength="2000"
    ></textarea>
    <div class="char-count"><span id="charCount">0</span> / 2000</div>

    <div class="presets">
      <span class="preset" onclick="setPreset(this)">The product exceeded every expectation</span>
      <span class="preset" onclick="setPreset(this)">Absolute disaster — never buying again</span>
      <span class="preset" onclick="setPreset(this)">Chainguard images are a game changer for supply chain security</span>
      <span class="preset" onclick="setPreset(this)">I waited three hours and it still didn't arrive</span>
      <span class="preset" onclick="setPreset(this)">The team delivered an outstanding result under pressure</span>
    </div>

    <button id="analyzeBtn" onclick="analyze()">Analyse Sentiment</button>

    <div id="result"></div>
  </div>

  <div class="info-strip">
    <div class="info-pill">
      <div class="pill-label">Model</div>
      <div class="pill-value">DistilBERT SST-2</div>
    </div>
    <div class="info-pill">
      <div class="pill-label">Base image</div>
      <div class="pill-value">cgr.dev/chainguard/pytorch</div>
    </div>
    <div class="info-pill">
      <div class="pill-label">Inference</div>
      <div class="pill-value">100% local · CPU/GPU</div>
    </div>
    <div class="info-pill">
      <div class="pill-label">Chat UI</div>
      <div class="pill-value"><a href="http://localhost:3000" style="color:var(--accent-light)">Open WebUI →</a></div>
    </div>
  </div>

  <script>
    const textarea = document.getElementById('inputText');
    const charCount = document.getElementById('charCount');
    const btn       = document.getElementById('analyzeBtn');
    const resultEl  = document.getElementById('result');

    textarea.addEventListener('input', () => {
      charCount.textContent = textarea.value.length;
    });

    function setPreset(el) {
      textarea.value = el.textContent.trim();
      charCount.textContent = textarea.value.length;
      analyze();
    }

    async function analyze() {
      const text = textarea.value.trim();
      if (!text) return;

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Analysing…';
      resultEl.innerHTML = '';

      try {
        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Server error');
        }

        const d = await res.json();
        renderResult(d);
      } catch (e) {
        resultEl.innerHTML = `<div class="error">⚠️ ${e.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Analyse Sentiment';
      }
    }

    function renderResult(d) {
      resultEl.innerHTML = `
        <div class="result-card">
          <div class="result-header">
            <div class="emoji">${d.emoji}</div>
            <div class="label-group">
              <div class="sentiment-label" style="color:${d.color}">${d.label}</div>
              <div class="confidence-text">${d.confidence_pct}% confidence</div>
            </div>
            <div class="latency">${d.latency_ms} ms</div>
          </div>

          <div class="bar-track">
            <div
              class="bar-fill"
              style="width:0%;background:${d.color}"
              id="bar"
            ></div>
          </div>
          <div class="bar-labels">
            <span>Negative</span>
            <span>Positive</span>
          </div>
        </div>
      `;

      // Animate bar after paint
      requestAnimationFrame(() => {
        const pct = d.label === 'POSITIVE' ? d.confidence_pct : 100 - d.confidence_pct;
        document.getElementById('bar').style.width = pct + '%';
      });
    }

    // Ctrl/Cmd + Enter to submit
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
