#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/saluki-uv-cache}"
exec uv run ty check
