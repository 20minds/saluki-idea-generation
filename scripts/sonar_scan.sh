#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/saluki-uv-cache}"

if [[ -z "${SONAR_HOST_URL:-}" || -z "${SONAR_TOKEN:-}" ]]; then
  echo "SonarQube scan is not configured: set SONAR_HOST_URL and SONAR_TOKEN."
  exit 1
fi

project_key="${SONAR_PROJECT_KEY:-saluki-idea-generation}"
project_name="${SONAR_PROJECT_NAME:-${project_key}}"
scanner_host_url="${SONAR_SCANNER_HOST_URL:-${SONAR_HOST_URL}}"
quality_gate_timeout="${SONAR_QUALITYGATE_TIMEOUT:-300}"

if [[ "${scanner_host_url}" == "http://localhost:9000" ]]; then
  scanner_host_url="http://[::1]:9000"
fi

if ! command -v sonar-scanner >/dev/null 2>&1; then
  echo "sonar-scanner is not installed or not on PATH."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH."
  exit 1
fi

uv run python -m pytest --cov=saluki --cov-report=xml:coverage.xml --cov-report=term-missing

exec sonar-scanner \
  -Dproject.settings=config/sonar-project.properties \
  -Dsonar.host.url="${scanner_host_url}" \
  -Dsonar.token="${SONAR_TOKEN}" \
  -Dsonar.projectKey="${project_key}" \
  -Dsonar.projectName="${project_name}" \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout="${quality_gate_timeout}"
