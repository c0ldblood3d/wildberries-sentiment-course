#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-wildberries-sentiment:smoke}"
CONTAINER="wb-sentiment-smoke-$$"
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker build -t "$IMAGE" .
docker run -d --name "$CONTAINER" -p 127.0.0.1:18000:8000 "$IMAGE"
for attempt in $(seq 1 30); do
  if curl --silent --fail http://127.0.0.1:18000/health >/dev/null; then break; fi
  sleep 1
done
curl --fail http://127.0.0.1:18000/health
curl --fail --silent http://127.0.0.1:18000/docs -o /dev/null
curl --fail -H 'Content-Type: application/json' \
  -d '{"text":"Отличное качество, очень нравится!"}' http://127.0.0.1:18000/predict
code="$(curl --silent -o /tmp/wb-sentiment-error.json -w '%{http_code}' -H 'Content-Type: application/json' -d '{"text":"   "}' http://127.0.0.1:18000/predict)"
test "$code" = "422"
