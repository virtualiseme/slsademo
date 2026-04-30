"""
Security Posture Comparison Dashboard

Serves a side-by-side visual comparison of CVE findings between a standard
upstream Docker image and the Chainguard hardened equivalent.

Data source priority:
  1. AWS S3  — real scan results uploaded by the Security Comparison workflow
  2. Local   — comparison-summary.json if mounted at /data/summary.json
  3. Demo    — bundled demo_data.json (realistic mock data, no AWS required)

Endpoints:
  GET  /          → dashboard HTML
  GET  /api/data  → comparison JSON
  GET  /health    → {"status": "ok"}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import botocore
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Security Posture Dashboard")

DEMO_DATA_PATH  = Path(__file__).parent / "demo_data.json"
LOCAL_DATA_PATH = Path("/data/summary.json")
S3_BUCKET       = os.environ.get("SBOM_BUCKET", "")
S3_KEY          = "comparison/latest/summary.json"
GRAFANA_URL     = os.environ.get("GRAFANA_URL", "")


def load_data() -> dict:
    # 1 — try S3
    if S3_BUCKET:
        try:
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
            obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
            data = json.loads(obj["Body"].read())
            data["_source"] = "s3"
            return data
        except (botocore.exceptions.ClientError, Exception):
            pass

    # 2 — try local mount
    if LOCAL_DATA_PATH.exists():
        data = json.loads(LOCAL_DATA_PATH.read_text())
        data["_source"] = "local"
        return data

    # 3 — bundled demo data
    data = json.loads(DEMO_DATA_PATH.read_text())
    data["_source"] = "demo"
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/data")
def api_data():
    data = load_data()
    if GRAFANA_URL:
        data["grafana_url"] = GRAFANA_URL
    return JSONResponse(data)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


# ─────────────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Security Posture — Chainguard vs Standard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#080812;
      --surface:#0f0f1f;
      --surface2:#16162e;
      --border:#252545;
      --text:#e2e8f0;
      --muted:#64748b;
      --red:#ef4444;
      --red-dim:rgba(239,68,68,.12);
      --amber:#f59e0b;
      --green:#22c55e;
      --green-dim:rgba(34,197,94,.12);
      --blue:#60a5fa;
      --purple:#a78bfa;
      --radius:14px;
    }
    body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh}

    /* ── Nav ─────────────────────────────────────────────── */
    nav{
      display:flex;align-items:center;justify-content:space-between;
      padding:.9rem 2rem;
      background:var(--surface);border-bottom:1px solid var(--border);
    }
    .nav-brand{display:flex;align-items:center;gap:.6rem;font-weight:700;font-size:1rem}
    .nav-meta{font-size:.75rem;color:var(--muted);text-align:right}
    .source-badge{
      display:inline-block;padding:.15rem .5rem;border-radius:99px;font-size:.65rem;
      font-weight:600;text-transform:uppercase;letter-spacing:.06em;
      background:rgba(96,165,250,.15);color:var(--blue);border:1px solid rgba(96,165,250,.3);
      margin-left:.4rem;
    }

    /* ── Hero ────────────────────────────────────────────── */
    .hero{
      text-align:center;padding:2.5rem 1rem 1.5rem;
      background:linear-gradient(180deg,rgba(167,139,250,.06) 0%,transparent 100%);
    }
    .hero-label{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem}
    .hero-number{
      font-size:clamp(3.5rem,8vw,6rem);font-weight:900;line-height:1;
      background:linear-gradient(135deg,#22c55e,#60a5fa);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    }
    .hero-sub{font-size:.9rem;color:var(--muted);margin-top:.5rem}

    /* ── Comparison grid ─────────────────────────────────── */
    .comparison{
      display:grid;grid-template-columns:1fr auto 1fr;gap:0;
      max-width:1200px;margin:0 auto;padding:1.5rem 1rem 2rem;
      align-items:start;
    }
    .panel{
      background:var(--surface);border:1px solid var(--border);
      border-radius:var(--radius);padding:1.75rem;
    }
    .panel.standard{border-top:3px solid var(--red)}
    .panel.chainguard{border-top:3px solid var(--green)}

    .panel-header{margin-bottom:1.5rem}
    .panel-tag{
      font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
      padding:.2rem .6rem;border-radius:99px;margin-bottom:.6rem;display:inline-block;
    }
    .standard .panel-tag{background:var(--red-dim);color:var(--red)}
    .chainguard .panel-tag{background:var(--green-dim);color:var(--green)}

    .panel-title{font-size:1.1rem;font-weight:700;margin-bottom:.2rem}
    .panel-image{font-size:.72rem;color:var(--muted);font-family:monospace}

    /* Total CVE count */
    .cve-total{
      text-align:center;padding:1.2rem 0 1rem;
      border-bottom:1px solid var(--border);margin-bottom:1.25rem;
    }
    .cve-total-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
    .cve-total-num{
      font-size:4rem;font-weight:900;line-height:1;
      display:block;margin:.2rem 0;
    }
    .standard .cve-total-num{color:var(--red)}
    .chainguard .cve-total-num{color:var(--green)}
    .pass-fail{
      display:inline-flex;align-items:center;gap:.35rem;
      font-size:.78rem;font-weight:600;padding:.25rem .7rem;
      border-radius:99px;margin-top:.4rem;
    }
    .pass{background:var(--green-dim);color:var(--green)}
    .fail{background:var(--red-dim);color:var(--red)}

    /* Chart */
    .chart-wrap{width:160px;height:160px;margin:0 auto 1.25rem}

    /* Severity bars */
    .sev-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem;font-size:.82rem}
    .sev-label{width:70px;color:var(--muted);flex-shrink:0}
    .sev-bar-track{flex:1;background:rgba(255,255,255,.05);border-radius:99px;height:6px;overflow:hidden}
    .sev-bar-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.16,1,.3,1)}
    .sev-count{width:36px;text-align:right;font-weight:600;font-size:.8rem}
    .c-crit{background:#ef4444}.c-high{background:#f97316}.c-med{background:#eab308}
    .c-low{background:#22c55e}.c-neg{background:#64748b}

    /* Score */
    .score-row{
      display:flex;align-items:center;justify-content:space-between;
      background:var(--surface2);border-radius:10px;padding:.85rem 1rem;margin-top:1rem;
    }
    .score-label{font-size:.8rem;color:var(--muted)}
    .score-val{font-size:1.5rem;font-weight:800}
    .score-val.low{color:var(--red)}
    .score-val.med{color:var(--amber)}
    .score-val.high{color:var(--green)}
    .pkg-count{font-size:.78rem;color:var(--muted);margin-top:.5rem;text-align:center}

    /* VS divider */
    .vs-col{
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:0 1.5rem;gap:.8rem;position:sticky;top:50%;
    }
    .vs-badge{
      background:var(--surface2);border:1px solid var(--border);
      border-radius:99px;width:48px;height:48px;
      display:flex;align-items:center;justify-content:center;
      font-weight:800;font-size:.85rem;color:var(--muted);
    }
    .delta-item{text-align:center}
    .delta-pct{font-size:1.4rem;font-weight:800;color:var(--green)}
    .delta-label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}

    /* ── Metrics strip ───────────────────────────────────── */
    .metrics{
      max-width:1200px;margin:0 auto;padding:0 1rem 3rem;
      display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.75rem;
    }
    .metric-card{
      background:var(--surface);border:1px solid var(--border);
      border-radius:10px;padding:1rem 1.25rem;
    }
    .metric-card-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem}
    .metric-card-value{font-size:1.25rem;font-weight:700}
    .metric-card-sub{font-size:.72rem;color:var(--muted);margin-top:.15rem}

    /* ── Recommendations ─────────────────────────────────── */
    .recs{
      max-width:1200px;margin:0 auto;padding:0 1rem 3rem;
    }
    .recs h2{font-size:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem}
    .rec-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.75rem}
    .rec-card{
      background:var(--surface);border:1px solid var(--border);
      border-radius:10px;padding:1.25rem;
    }
    .rec-card h3{font-size:.9rem;font-weight:700;margin-bottom:.4rem;display:flex;align-items:center;gap:.5rem}
    .rec-card p{font-size:.8rem;color:var(--muted);line-height:1.6}
    .rec-badge{
      font-size:.65rem;font-weight:700;padding:.15rem .45rem;border-radius:99px;
    }
    .rec-badge.aws{background:rgba(249,115,22,.15);color:#fb923c}
    .rec-badge.oss{background:rgba(96,165,250,.15);color:var(--blue)}
    .rec-badge.live{background:rgba(34,197,94,.15);color:var(--green)}

    /* ── Refresh button ──────────────────────────────────── */
    .refresh-btn{
      position:fixed;bottom:1.5rem;right:1.5rem;
      background:linear-gradient(135deg,#7c3aed,#4f46e5);
      border:none;border-radius:99px;color:#fff;
      font-size:.8rem;font-weight:600;padding:.6rem 1.2rem;cursor:pointer;
      box-shadow:0 4px 20px rgba(124,58,237,.4);transition:opacity .2s;
    }
    .refresh-btn:hover{opacity:.85}

    @keyframes countUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
    .anim{animation:countUp .6s ease both}
  </style>
</head>
<body>
  <nav>
    <div class="nav-brand">🔗 Chainguard Security Posture Dashboard</div>
    <div style="display:flex;align-items:center;gap:1rem">
      <a id="grafanaBtn" href="#" target="_blank" rel="noopener" style="display:none;
        background:linear-gradient(135deg,#f97316,#ef4444);border-radius:99px;
        color:#fff;font-size:.78rem;font-weight:700;padding:.4rem 1rem;text-decoration:none;
        letter-spacing:.03em;">📊 Open Grafana</a>
      <div class="nav-meta">
        Last scan: <span id="scanTime">—</span>
        <span class="source-badge" id="sourceLabel">loading</span>
      </div>
    </div>
  </nav>

  <div class="hero">
    <div class="hero-label">CVE Reduction vs Standard Upstream Image</div>
    <div class="hero-number" id="heroReduction">—</div>
    <div class="hero-sub" id="heroSub">Loading scan data…</div>
  </div>

  <div class="comparison">
    <!-- Standard panel -->
    <div class="panel standard" id="stdPanel">
      <div class="panel-header">
        <span class="panel-tag">Standard Upstream</span>
        <div class="panel-title">Python Official Image</div>
        <div class="panel-image" id="stdImage">python:3.11-slim</div>
      </div>
      <div class="cve-total">
        <div class="cve-total-label">Total CVEs</div>
        <span class="cve-total-num anim" id="stdTotal">—</span>
        <div><span class="pass-fail fail" id="stdGate">❌ Gate: FAIL</span></div>
      </div>
      <div class="chart-wrap"><canvas id="stdChart"></canvas></div>
      <div id="stdSevBars"></div>
      <div class="score-row">
        <span class="score-label">Security Score</span>
        <span class="score-val" id="stdScore">—</span>
      </div>
      <div class="pkg-count" id="stdPkgs">— packages</div>
    </div>

    <!-- VS column -->
    <div class="vs-col">
      <div class="vs-badge">VS</div>
      <div class="delta-item">
        <div class="delta-pct" id="deltaTotal">—</div>
        <div class="delta-label">Fewer CVEs</div>
      </div>
      <div class="delta-item">
        <div class="delta-pct" id="deltaCrit">—</div>
        <div class="delta-label">Critical↓</div>
      </div>
      <div class="delta-item">
        <div class="delta-pct" id="deltaPkg">—</div>
        <div class="delta-label">Fewer pkgs</div>
      </div>
    </div>

    <!-- Chainguard panel -->
    <div class="panel chainguard" id="cgPanel">
      <div class="panel-header">
        <span class="panel-tag">Chainguard Hardened</span>
        <div class="panel-title">Chainguard PyTorch</div>
        <div class="panel-image" id="cgImage">cgr.dev/chainguard-private/pytorch</div>
      </div>
      <div class="cve-total">
        <div class="cve-total-label">Total CVEs</div>
        <span class="cve-total-num anim" id="cgTotal">—</span>
        <div><span class="pass-fail" id="cgGate">✅ Gate: PASS</span></div>
      </div>
      <div class="chart-wrap"><canvas id="cgChart"></canvas></div>
      <div id="cgSevBars"></div>
      <div class="score-row">
        <span class="score-label">Security Score</span>
        <span class="score-val" id="cgScore">—</span>
      </div>
      <div class="pkg-count" id="cgPkgs">— packages</div>
    </div>
  </div>

  <!-- Key metrics -->
  <div class="metrics" id="metricsStrip"></div>

  <!-- Recommendations -->
  <div class="recs">
    <h2>Observability Recommendations</h2>
    <div class="rec-grid">
      <div class="rec-card">
        <h3><span class="rec-badge live">Live</span> This Dashboard</h3>
        <p>Reads real Grype scan results from S3, uploaded by the Security Comparison pipeline on every build. Run <code>workflow_dispatch</code> to refresh during a live demo.</p>
      </div>
      <div class="rec-card">
        <h3><span class="rec-badge aws">AWS</span> Inspector v2 + Security Hub</h3>
        <p>Enable Inspector v2 in your account — it automatically scans every ECR image on push. Findings aggregate in Security Hub with NIST/CIS compliance scoring and trend analysis over time.</p>
      </div>
      <div class="rec-card">
        <h3><span class="rec-badge aws">AWS</span> CloudWatch Dashboard</h3>
        <p>Push Grype CVE counts as CloudWatch custom metrics from the pipeline. Build a dashboard with alarms — alert when CRITICAL CVEs appear in any image pushed to ECR.</p>
      </div>
      <div class="rec-card" id="grafanaCard">
        <h3><span class="rec-badge oss" id="grafanaBadge">OSS</span> Amazon Managed Grafana</h3>
        <p id="grafanaDesc">Deploy the AMG workspace via Terraform (<code>enable_grafana = true</code>), then run <code>scripts/grafana-provision.sh</code> to upload the CVE comparison dashboard. Set <code>GRAFANA_URL</code> in GitHub Actions to link it here.</p>
      </div>
    </div>
  </div>

  <button class="refresh-btn" onclick="loadData()">↻ Refresh Data</button>

  <script>
    const SEV_COLORS = {
      critical:'#ef4444', high:'#f97316', medium:'#eab308',
      low:'#22c55e', negligible:'#64748b'
    };
    const SEV_FILL = {
      critical:'c-crit', high:'c-high', medium:'c-med', low:'c-low', negligible:'c-neg'
    };
    let stdChart, cgChart;

    async function loadData() {
      const res = await fetch('/api/data');
      const d   = await res.json();
      render(d);
    }

    function fmt(n) { return n === undefined ? '—' : n.toLocaleString(); }
    function pct(n)  { return n === undefined ? '—' : n.toFixed(1) + '%'; }

    function scoreClass(s) {
      if (s >= 80) return 'high';
      if (s >= 50) return 'med';
      return 'low';
    }

    function sevBars(el, data, maxVal) {
      const sevs = ['critical','high','medium','low','negligible'];
      el.innerHTML = sevs.map(s => {
        const count = data[s] ?? 0;
        const w = maxVal > 0 ? Math.max(2, (count / maxVal) * 100) : 2;
        return `<div class="sev-row">
          <span class="sev-label">${s.charAt(0).toUpperCase()+s.slice(1)}</span>
          <div class="sev-bar-track">
            <div class="sev-bar-fill ${SEV_FILL[s]}" style="width:${count>0?w:0}%"></div>
          </div>
          <span class="sev-count">${count}</span>
        </div>`;
      }).join('');
    }

    function donut(canvas, data, label) {
      const sevs   = ['critical','high','medium','low','negligible'];
      const counts = sevs.map(s => data[s] ?? 0);
      const total  = counts.reduce((a,b)=>a+b, 0);
      const cfg = {
        type: 'doughnut',
        data: {
          labels: sevs.map(s=>s.charAt(0).toUpperCase()+s.slice(1)),
          datasets:[{
            data: total > 0 ? counts : [1],
            backgroundColor: total > 0
              ? sevs.map(s => SEV_COLORS[s])
              : ['rgba(255,255,255,.05)'],
            borderWidth: 0,
            hoverOffset: 4,
          }]
        },
        options:{
          responsive:true, maintainAspectRatio:false, cutout:'72%',
          plugins:{
            legend:{display:false},
            tooltip:{enabled: total > 0},
          }
        }
      };
      if (canvas._chart) canvas._chart.destroy();
      const c = new Chart(canvas, cfg);
      canvas._chart = c;
      return c;
    }

    function render(d) {
      // Hero
      document.getElementById('heroReduction').textContent =
        (d.delta?.total_reduction_pct ?? 0).toFixed(1) + '%';
      document.getElementById('heroSub').textContent =
        `${d.chainguard?.total ?? 0} CVEs in Chainguard vs ${d.standard?.total ?? 0} in standard upstream`;

      // Timestamps
      const ts = d.generated_at ? new Date(d.generated_at).toLocaleString() : '—';
      document.getElementById('scanTime').textContent = ts;
      const src = d._source ?? 'unknown';
      const srcEl = document.getElementById('sourceLabel');
      srcEl.textContent = src === 'demo' ? 'demo data' : src === 's3' ? 'live · S3' : 'local';

      // Standard panel
      document.getElementById('stdImage').textContent = d.standard?.image ?? '';
      document.getElementById('stdTotal').textContent = fmt(d.standard?.total);
      document.getElementById('stdScore').textContent = (d.standard?.score ?? 0) + '/100';
      document.getElementById('stdScore').className =
        'score-val ' + scoreClass(d.standard?.score ?? 0);
      document.getElementById('stdPkgs').textContent =
        (d.standard?.package_count ?? 0) + ' packages scanned';
      const stdGate = document.getElementById('stdGate');
      if (d.standard?.scan_passed) {
        stdGate.textContent = '✅ Gate: PASS'; stdGate.className='pass-fail pass';
      } else {
        stdGate.textContent = '❌ Gate: FAIL'; stdGate.className='pass-fail fail';
      }
      sevBars(document.getElementById('stdSevBars'), d.standard ?? {}, d.standard?.total ?? 1);
      donut(document.getElementById('stdChart'), d.standard ?? {});

      // Chainguard panel
      document.getElementById('cgImage').textContent = d.chainguard?.image ?? '';
      document.getElementById('cgTotal').textContent = fmt(d.chainguard?.total);
      document.getElementById('cgScore').textContent = (d.chainguard?.score ?? 0) + '/100';
      document.getElementById('cgScore').className =
        'score-val ' + scoreClass(d.chainguard?.score ?? 0);
      document.getElementById('cgPkgs').textContent =
        (d.chainguard?.package_count ?? 0) + ' packages scanned';
      const cgGate = document.getElementById('cgGate');
      if (d.chainguard?.scan_passed) {
        cgGate.textContent = '✅ Gate: PASS'; cgGate.className='pass-fail pass';
      } else {
        cgGate.textContent = '❌ Gate: FAIL'; cgGate.className='pass-fail fail';
      }
      sevBars(document.getElementById('cgSevBars'), d.chainguard ?? {}, d.standard?.total ?? 1);
      donut(document.getElementById('cgChart'), d.chainguard ?? {});

      // Delta column
      document.getElementById('deltaTotal').textContent = pct(d.delta?.total_reduction_pct);
      document.getElementById('deltaCrit').textContent  = pct(d.delta?.critical_reduction_pct);
      document.getElementById('deltaPkg').textContent   = pct(d.delta?.package_reduction_pct);

      // Grafana button
      const grafanaBtn = document.getElementById('grafanaBtn');
      if (d.grafana_url) {
        grafanaBtn.href = d.grafana_url;
        grafanaBtn.style.display = 'inline-block';
      }

      // Engineering hours saved
      const hrs     = d.delta?.engineering_hours_saved ?? 0;
      const weeks   = (hrs / 40).toFixed(1);
      const hrsText = hrs > 0 ? `${hrs.toLocaleString()} hrs` : '—';

      // Image sizes
      const stdMB = d.standard?.size_mb;
      const cgMB  = d.chainguard?.size_mb;
      const sizePct = d.delta?.size_reduction_pct;

      const sizeValue = (stdMB && cgMB)
        ? `${cgMB} MB vs ${stdMB} MB`
        : '—';
      const sizeSub = sizePct
        ? `${sizePct}% smaller (${stdMB - cgMB} MB saved)`
        : 'Run comparison workflow to capture sizes';

      // Metrics strip
      document.getElementById('metricsStrip').innerHTML = [
        { label:'Engineering hours saved',  value:hrsText,                                         sub:`≈ ${weeks} engineering weeks per build cycle` },
        { label:'Score improvement',        value:`+${d.delta?.score_improvement ?? 0} pts`,       sub:'Security score delta' },
        { label:'Critical CVEs removed',    value:d.standard?.critical ?? 0,                       sub:'All eliminated in Chainguard' },
        { label:'Package reduction',        value:pct(d.delta?.package_reduction_pct),             sub:'Fewer packages = smaller attack surface' },
        { label:'Image size',               value:sizeValue,                                       sub:sizeSub },
        { label:'Data source',              value:(d._source==='s3'?'Live S3':'Demo data'),        sub:'Run comparison workflow for live results' },
      ].map(m=>`<div class="metric-card">
        <div class="metric-card-label">${m.label}</div>
        <div class="metric-card-value">${m.value}</div>
        <div class="metric-card-sub">${m.sub}</div>
      </div>`).join('');

      // Update Grafana recommendation card when live
      if (d.grafana_url) {
        const badge = document.getElementById('grafanaBadge');
        const desc  = document.getElementById('grafanaDesc');
        if (badge) { badge.textContent = 'Live'; badge.className = 'rec-badge live'; }
        if (desc)  {
          desc.innerHTML = `AMG workspace is live. CVE trends, severity breakdown, score history, image sizes, and engineering hours are all streaming from CloudWatch. ` +
            `<a href="${d.grafana_url}" target="_blank" rel="noopener" style="color:var(--green)">Open dashboard →</a>`;
        }
      }
    }

    loadData();
    // Auto-refresh every 60 seconds when live data is connected
    setInterval(loadData, 60000);
  </script>
</body>
</html>"""
