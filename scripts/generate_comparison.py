#!/usr/bin/env python3
"""
Parse Grype JSON scan results for two images and emit a comparison-summary.json
plus a GitHub Actions job summary (Markdown). Optionally publish CloudWatch metrics.

Usage:
    python generate_comparison.py \
        --standard  standard-scan.json \
        --chainguard chainguard-scan.json \
        --standard-image  "python:3.11-slim" \
        --chainguard-image "cgr.dev/chainguard-private/pytorch:latest-dev" \
        --standard-size-mb 168 \
        --chainguard-size-mb 104 \
        --out comparison-summary.json \
        --github-summary \
        --publish-cloudwatch          # writes to SecurityDashboard namespace
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone


SEVERITIES = ["critical", "high", "medium", "low", "negligible"]

# Engineering effort per CVE by severity (industry consensus, hours)
HOURS_PER_CVE = {
    "critical":   8.0,   # security triage + patch + test + emergency deploy
    "high":       4.0,   # patch + PR + staged rollout
    "medium":     2.0,   # scheduled patch cycle
    "low":        0.5,   # batch remediation
    "negligible": 0.0,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def count_matches(grype_json: dict) -> dict:
    counts = {s: 0 for s in SEVERITIES}
    for match in grype_json.get("matches", []):
        sev = match.get("vulnerability", {}).get("severity", "unknown").lower()
        if sev in counts:
            counts[sev] += 1
    counts["total"] = sum(counts.values())
    return counts


def security_score(counts: dict) -> int:
    """0–100 score. Deduct per CVE weighted by severity."""
    score = 100
    score -= counts.get("critical", 0) * 20
    score -= counts.get("high", 0) * 5
    score -= counts.get("medium", 0) * 1
    score -= counts.get("low", 0) * 0.2
    return max(0, round(score))


def reduction_pct(a: int, b: int) -> float:
    if a == 0:
        return 0.0
    return round((a - b) / a * 100, 1)


def engineering_hours_saved(std: dict, cg: dict) -> float:
    """Calculate total engineering hours saved by eliminating CVEs."""
    hours = 0.0
    for sev, hrs in HOURS_PER_CVE.items():
        delta = max(0, std.get(sev, 0) - cg.get(sev, 0))
        hours += delta * hrs
    return round(hours, 1)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_summary(args) -> dict:
    std_raw = load(args.standard)
    cg_raw  = load(args.chainguard)

    std  = count_matches(std_raw)
    cg   = count_matches(cg_raw)

    std_pkgs = len({m["artifact"]["name"] for m in std_raw.get("matches", [])})
    cg_pkgs  = len({m["artifact"]["name"] for m in cg_raw.get("matches", [])})

    std_size = float(args.standard_size_mb)   if args.standard_size_mb  else None
    cg_size  = float(args.chainguard_size_mb) if args.chainguard_size_mb else None

    hours_saved = engineering_hours_saved(std, cg)

    std_entry = {
        "image":         args.standard_image,
        "scan_passed":   std["critical"] == 0 and std["high"] == 0,
        "package_count": std_pkgs,
        "score":         security_score(std),
        **std,
    }
    if std_size is not None:
        std_entry["size_mb"] = std_size

    cg_entry = {
        "image":         args.chainguard_image,
        "scan_passed":   cg["critical"] == 0,
        "package_count": cg_pkgs,
        "score":         security_score(cg),
        **cg,
    }
    if cg_size is not None:
        cg_entry["size_mb"] = cg_size

    delta = {
        "total_reduction_pct":    reduction_pct(std["total"],    cg["total"]),
        "critical_reduction_pct": reduction_pct(std["critical"], cg["critical"]),
        "high_reduction_pct":     reduction_pct(std["high"],     cg["high"]),
        "package_reduction_pct":  reduction_pct(std_pkgs,        cg_pkgs),
        "score_improvement":      security_score(cg) - security_score(std),
        "engineering_hours_saved": hours_saved,
    }
    if std_size and cg_size:
        delta["size_reduction_pct"] = reduction_pct(int(std_size), int(cg_size))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "standard":    std_entry,
        "chainguard":  cg_entry,
        "delta":       delta,
    }


def write_github_summary(summary: dict):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return

    d   = summary["delta"]
    std = summary["standard"]
    cg  = summary["chainguard"]

    size_row = ""
    if std.get("size_mb") and cg.get("size_mb"):
        size_row = f"| **Image size** | {std['size_mb']} MB | {cg['size_mb']} MB | {d.get('size_reduction_pct', 0)}% smaller |\n"

    hours = d.get("engineering_hours_saved", 0)
    weeks = round(hours / 40, 1)

    md = f"""## Security Comparison — Standard vs Chainguard

| | Standard (`{std['image']}`) | Chainguard (`{cg['image']}`) | Reduction |
|---|---|---|---|
| **Total CVEs** | {std['total']} | {cg['total']} | **{d['total_reduction_pct']}%** |
| Critical | {std['critical']} | {cg['critical']} | {d['critical_reduction_pct']}% |
| High | {std['high']} | {cg['high']} | {d['high_reduction_pct']}% |
| Medium | {std['medium']} | {cg['medium']} | — |
| Low | {std['low']} | {cg['low']} | — |
| **Packages** | {std['package_count']} | {cg['package_count']} | {d['package_reduction_pct']}% |
| **Security Score** | {std['score']}/100 | {cg['score']}/100 | +{d['score_improvement']} pts |
| **Build Gate** | {'✅ Pass' if std['scan_passed'] else '❌ Fail'} | {'✅ Pass' if cg['scan_passed'] else '❌ Fail'} | |
{size_row}

### 🕐 Engineering Hours Saved
**{hours} hours** ({weeks} engineering weeks) — estimated remediation effort eliminated per build cycle.

> Generated at {summary['generated_at']}
"""
    with open(path, "a") as f:
        f.write(md)
    log.info("GitHub Actions job summary written.")


def publish_cloudwatch(summary: dict, region: str):
    """Publish comparison metrics to CloudWatch SecurityDashboard namespace."""
    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=region)
    except ImportError:
        log.error("boto3 not available — skipping CloudWatch publish")
        return

    std   = summary["standard"]
    cg    = summary["chainguard"]
    delta = summary["delta"]

    metric_data: list[dict] = []

    # CVE counts by severity per image type
    for sev in SEVERITIES:
        for image_type, data in [("Standard", std), ("Chainguard", cg)]:
            metric_data.append({
                "MetricName": "CVECount",
                "Dimensions": [
                    {"Name": "ImageType", "Value": image_type},
                    {"Name": "Severity",  "Value": sev.capitalize()},
                ],
                "Value": float(data.get(sev, 0)),
                "Unit": "Count",
            })

    # Total CVEs
    for image_type, data in [("Standard", std), ("Chainguard", cg)]:
        metric_data.append({
            "MetricName": "TotalCVEs",
            "Dimensions": [{"Name": "ImageType", "Value": image_type}],
            "Value": float(data.get("total", 0)),
            "Unit": "Count",
        })

    # Security scores
    for image_type, data in [("Standard", std), ("Chainguard", cg)]:
        metric_data.append({
            "MetricName": "SecurityScore",
            "Dimensions": [{"Name": "ImageType", "Value": image_type}],
            "Value": float(data.get("score", 0)),
            "Unit": "Count",
        })

    # Image sizes (if captured)
    for image_type, data in [("Standard", std), ("Chainguard", cg)]:
        if data.get("size_mb"):
            metric_data.append({
                "MetricName": "ImageSizeMB",
                "Dimensions": [{"Name": "ImageType", "Value": image_type}],
                "Value": float(data["size_mb"]),
                "Unit": "Count",
            })

    # Package counts
    for image_type, data in [("Standard", std), ("Chainguard", cg)]:
        metric_data.append({
            "MetricName": "PackageCount",
            "Dimensions": [{"Name": "ImageType", "Value": image_type}],
            "Value": float(data.get("package_count", 0)),
            "Unit": "Count",
        })

    # Engineering hours saved (single scalar, no dimension)
    if delta.get("engineering_hours_saved"):
        metric_data.append({
            "MetricName": "EngineeringHoursSaved",
            "Dimensions": [],
            "Value": float(delta["engineering_hours_saved"]),
            "Unit": "Count",
        })

    # Batch in groups of 20 (CloudWatch API limit)
    published = 0
    for i in range(0, len(metric_data), 20):
        cw.put_metric_data(
            Namespace="SecurityDashboard",
            MetricData=metric_data[i : i + 20],
        )
        published += len(metric_data[i : i + 20])

    log.info("Published %d metrics to CloudWatch namespace 'SecurityDashboard'", published)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard",             required=True,  help="Path to standard image Grype JSON")
    p.add_argument("--chainguard",           required=True,  help="Path to Chainguard image Grype JSON")
    p.add_argument("--standard-image",       default="python:3.11-slim")
    p.add_argument("--chainguard-image",     default="cgr.dev/chainguard-private/pytorch:latest-dev")
    p.add_argument("--standard-size-mb",     default=None,   help="Compressed image size in MB (from docker inspect)")
    p.add_argument("--chainguard-size-mb",   default=None,   help="Compressed image size in MB (from docker inspect)")
    p.add_argument("--out",                  default="comparison-summary.json")
    p.add_argument("--github-summary",       action="store_true", help="Append Markdown table to GITHUB_STEP_SUMMARY")
    p.add_argument("--publish-cloudwatch",   action="store_true", help="Push metrics to CloudWatch SecurityDashboard namespace")
    p.add_argument("--aws-region",           default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    args = p.parse_args()

    summary = build_summary(args)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    std = summary["standard"]
    cg  = summary["chainguard"]
    d   = summary["delta"]

    log.info("Comparison summary → %s", args.out)
    log.info("  Standard:   %d CVEs  (score %d/100)  %s",
             std["total"], std["score"],
             f"{std['size_mb']} MB" if std.get("size_mb") else "")
    log.info("  Chainguard: %d CVEs  (score %d/100)  %s",
             cg["total"], cg["score"],
             f"{cg['size_mb']} MB" if cg.get("size_mb") else "")
    log.info("  Reduction:  %s%%   Engineering hours saved: %s",
             d["total_reduction_pct"], d.get("engineering_hours_saved", "n/a"))

    if args.github_summary:
        write_github_summary(summary)

    if args.publish_cloudwatch:
        publish_cloudwatch(summary, args.aws_region)


if __name__ == "__main__":
    main()
