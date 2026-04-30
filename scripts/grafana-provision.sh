#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# grafana-provision.sh
#
# Upload the CVE comparison dashboard JSON to an Amazon Managed Grafana (AMG)
# workspace using the Grafana HTTP API.
#
# Usage:
#   ./scripts/grafana-provision.sh <workspace-id> <aws-region>
#
# Example:
#   ./scripts/grafana-provision.sh g-abc1234567 ap-southeast-2
#
# Prerequisites:
#   - AWS CLI configured with credentials that have grafana:CreateWorkspaceApiKey
#   - jq installed
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE_ID="${1:?Usage: $0 <workspace-id> <aws-region>}"
AWS_REGION="${2:-ap-southeast-2}"
DASHBOARD_FILE="$(dirname "$0")/../services/security-dashboard/grafana/cve-comparison.json"

echo "→ Creating temporary Grafana API key..."
KEY_RESPONSE=$(aws grafana create-workspace-api-key \
  --workspace-id "$WORKSPACE_ID" \
  --key-name "provisioner-$(date +%s)" \
  --key-role ADMIN \
  --seconds-to-live 300 \
  --region "$AWS_REGION" \
  --output json)

API_KEY=$(echo "$KEY_RESPONSE" | jq -r '.key')
ENDPOINT=$(echo "$KEY_RESPONSE" | jq -r '.workspaceId' | xargs -I{} \
  aws grafana describe-workspace --workspace-id {} --region "$AWS_REGION" \
  --query 'workspace.endpoint' --output text)

echo "→ Workspace endpoint: https://${ENDPOINT}"

# Wrap the dashboard JSON in the Grafana import format
PAYLOAD=$(jq -n \
  --slurpfile dashboard "$DASHBOARD_FILE" \
  '{
    dashboard: $dashboard[0],
    overwrite: true,
    folderId: 0,
    inputs: [
      {
        name: "DS_CLOUDWATCH",
        type: "datasource",
        pluginId: "cloudwatch",
        value: "CloudWatch"
      }
    ]
  }')

echo "→ Uploading dashboard..."
HTTP_STATUS=$(curl -s -o /tmp/grafana-import-response.json -w "%{http_code}" \
  -X POST "https://${ENDPOINT}/api/dashboards/import" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if [ "$HTTP_STATUS" -eq 200 ]; then
  DASHBOARD_URL=$(jq -r '.importedUrl' /tmp/grafana-import-response.json)
  echo "✅ Dashboard uploaded successfully."
  echo "   URL: https://${ENDPOINT}${DASHBOARD_URL}"
  echo ""
  echo "Add this as a GitHub Actions variable:"
  echo "  GRAFANA_URL = https://${ENDPOINT}${DASHBOARD_URL}"
else
  echo "❌ Upload failed (HTTP ${HTTP_STATUS}):"
  cat /tmp/grafana-import-response.json
  exit 1
fi

# Clean up the temporary API key
aws grafana delete-workspace-api-key \
  --workspace-id "$WORKSPACE_ID" \
  --key-name "$(echo "$KEY_RESPONSE" | jq -r '.keyName')" \
  --region "$AWS_REGION" 2>/dev/null || true

echo "→ Temporary API key deleted."
