#!/usr/bin/env bash
# Stop and remove the serve container. Weights and caches stay — ./run.sh brings it back fast.
set -euo pipefail
# the container runs as root — hand ./cache back to the host user before it goes
docker exec lfm25-24b-a2b chown -R "$(id -u):$(id -g)" /cache >/dev/null 2>&1 || true
docker rm -f lfm25-24b-a2b >/dev/null 2>&1 && echo "✓ stopped" || echo "· not running"
