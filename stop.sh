#!/usr/bin/env bash
# Stop and remove the serve container. Weights and caches stay — ./run.sh brings it back fast.
set -euo pipefail
# the container runs as root — hand ./cache back to the host user before it goes
docker exec lfm25-24b-a2b chown -R "$(id -u):$(id -g)" /cache >/dev/null 2>&1 || true
# SIGTERM first: vLLM shuts its engine down and releases the GPU cleanly (a hard kill leaves the driver to tear the
# context down mid-flight — on an eGPU link that has dropped the card); -t 30 = then SIGKILL
docker stop -t 30 lfm25-24b-a2b >/dev/null 2>&1 || true
docker rm -f lfm25-24b-a2b >/dev/null 2>&1 && echo "✓ stopped" || echo "· not running"
