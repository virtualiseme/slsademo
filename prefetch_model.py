"""
Run at Docker image build time to download and cache the model weights.

Baking the weights into the image layer means:
  - Container startup is immediate (no download on first request)
  - TRANSFORMERS_OFFLINE=1 is safe — weights are always present
  - The model version is pinned to the image digest (reproducible)
"""
import os
from transformers import pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Pre-fetching model: {MODEL}", flush=True)
print(f"Cache directory:    {os.environ.get('HF_HOME', '~/.cache/huggingface')}", flush=True)

pipe = pipeline("text-classification", model=MODEL)

# Smoke test — verify the model loads and produces a sensible result
sample = "Chainguard images have zero CVEs and are already signed."
result = pipe(sample)[0]
print(f"Smoke test → label={result['label']}, score={result['score']:.4f}", flush=True)
assert result["label"] in ("POSITIVE", "NEGATIVE"), "Unexpected model output"

print("Model cached and verified successfully.", flush=True)
