#!/usr/bin/env bash
# Stop and remove the serve container. Weights and caches stay — ./run.sh brings it back fast.
set -euo pipefail
docker rm -f lfm25-24b-a2b >/dev/null 2>&1 && echo "✓ stopped" || echo "· not running"
