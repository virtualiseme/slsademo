# Chainguard AI Demo Stack

A self-contained AI demo using exclusively Chainguard hardened images.
Three services, one command, runs on any laptop with Docker.

```
http://localhost:3000   →  Open WebUI        (ChatGPT-style LLM chat)
http://localhost:8000   →  Sentiment API     (DistilBERT, visual UI)
http://localhost:11434  →  Ollama API        (LLM inference REST endpoint)
```

---

## Quick Start

```bash
cd demo

# 1. Copy and (optionally) edit the env file
cp .env.example .env

# 2. Start everything — first run pulls images and builds the sentiment service
docker compose up -d --build

# 3. Watch progress (model download can take a few minutes on first run)
docker compose logs -f model-init

# 4. Open the UIs once healthy
open http://localhost:3000   # Open WebUI
open http://localhost:8000   # Sentiment demo
```

> **First run note:** `model-init` pulls the LLM model (~2.3 GB for phi3:mini).
> Subsequent starts use the cached `chainguard-demo-ollama` volume — instant.

---

## Images Used

| Service | Chainguard Image | Purpose |
|---|---|---|
| `ollama` | `cgr.dev/chainguard/ollama:latest` | LLM inference server |
| `model-init` | `cgr.dev/chainguard/ollama:latest-dev` | One-shot model pull job |
| `openwebui` | `cgr.dev/chainguard/openwebui:latest` | Browser chat UI |
| `sentiment-api` | `cgr.dev/chainguard/pytorch:latest-dev` | DistilBERT inference + web UI |

All Chainguard images are:
- Distroless / minimal — dramatically reduced CVE surface
- Pre-signed with Cosign (verify with `cosign verify cgr.dev/chainguard/ollama:latest`)
- Shipped with an SBOM (inspect with `cosign verify-attestation --type spdxjson ...`)
- Rebuilt daily against upstream patches

---

## Choosing a Model

Edit `.env` to change the default model:

```env
# Tiny — works on any laptop, 4 GB RAM
OLLAMA_MODEL=tinyllama

# Default — good balance of quality and speed on CPU
OLLAMA_MODEL=phi3:mini

# Better quality — recommended with 8+ GB RAM
OLLAMA_MODEL=llama3.2:1b
```

Pull additional models interactively after the stack is up:

```bash
docker compose exec model-init ollama pull mistral:7b
```

---

## GPU Acceleration (Nvidia)

```bash
# Prerequisite: nvidia-container-toolkit installed on the host
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The GPU override:
- Passes all Nvidia GPUs to the Ollama service
- Defaults to `llama3.2:8b` for a noticeably better chat experience
- Passes GPU 0 to the sentiment API (inference latency drops from ~200 ms to ~20 ms)

---

## Verify Supply Chain Integrity

Every Chainguard image is signed. You can verify this before running:

```bash
# Verify Ollama image signature
cosign verify \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  --certificate-identity-regexp=https://github.com/chainguard-images/.* \
  cgr.dev/chainguard/ollama:latest

# Inspect the SBOM attached to the PyTorch image
cosign verify-attestation \
  --type spdxjson \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  --certificate-identity-regexp=https://github.com/chainguard-images/.* \
  cgr.dev/chainguard/pytorch:latest-dev \
  | jq '.payload | @base64d | fromjson | .packages | length'
```

---

## Service Architecture

```
Browser
  │
  ├── :3000 → openwebui ──────────────────► ollama :11434
  │                                              ▲
  │                                         model-init (exits after pull)
  │
  └── :8000 → sentiment-api (FastAPI)
                  └── DistilBERT (baked into image at build time)
```

The sentiment-api builds in two stages (see [Dockerfile](services/sentiment-api/Dockerfile)):
1. **Builder** — installs deps and downloads model weights at build time
2. **Runtime** — copies deps + cached weights, runs with `TRANSFORMERS_OFFLINE=1`

This means the sentiment container starts in ~2 seconds with no network calls.

---

## Stopping / Cleaning Up

```bash
# Stop containers (preserves model data in volumes)
docker compose down

# Stop and remove all volumes (deletes downloaded models — re-download required)
docker compose down -v
```
