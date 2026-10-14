#!/usr/bin/env bash
# One-shot setup on the SSH host (no root needed):
#   bash setup.sh
# Installs cloudflared to ~/bin, starts the relay and a quick tunnel in tmux/nohup,
# and prints RELAY_URL and RELAY_TOKEN to put in the Claude environment settings.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/bin" "$HOME/work"

case "$(uname -m)" in
  x86_64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac
if [ ! -x "$HOME/bin/cloudflared" ]; then
  curl -fL -o "$HOME/bin/cloudflared" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH"
  chmod +x "$HOME/bin/cloudflared"
fi

if [ -f "$HOME/.relay_token" ]; then TOKEN=$(cat "$HOME/.relay_token"); else
  TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')
  (umask 077; echo "$TOKEN" > "$HOME/.relay_token")
fi

pkill -f "$DIR/server.py" 2>/dev/null || true
pkill -f "cloudflared tunnel --url" 2>/dev/null || true
RELAY_TOKEN="$TOKEN" RELAY_ROOT="$HOME/work" nohup python3 "$DIR/server.py" >"$HOME/.relay.out" 2>&1 &
sleep 1
nohup "$HOME/bin/cloudflared" tunnel --url http://127.0.0.1:8787 >"$HOME/.cloudflared.out" 2>&1 &

for i in $(seq 1 30); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$HOME/.cloudflared.out" | head -1 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -n "${URL:-}" ] || { echo "no tunnel URL; see ~/.cloudflared.out"; exit 1; }
echo
echo "RELAY_URL=$URL"
echo "RELAY_TOKEN=$TOKEN"
echo "(token also saved in ~/.relay_token; do not paste it anywhere public)"
