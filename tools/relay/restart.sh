#!/usr/bin/env bash
# Run from the relay folder: bash restart.sh
# New token + restart the relay server only (the tunnel keeps its URL).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/work"
TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')
(umask 077; echo "$TOKEN" > "$HOME/.relay_token")
pkill -f "$DIR/server.py" 2>/dev/null || pkill -f "python3 server.py" 2>/dev/null || true
sleep 1
RELAY_TOKEN="$TOKEN" RELAY_ROOT="$HOME/work" nohup python3 "$DIR/server.py" >"$HOME/.relay.out" 2>&1 &
sleep 1
curl -s -o /dev/null -w "local check (expect 401): %{http_code}\n" -X POST http://127.0.0.1:8787/exec -d '{}'
echo
echo "RELAY_TOKEN=$TOKEN"
