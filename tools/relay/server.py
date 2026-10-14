#!/usr/bin/env python3
"""Minimal token-authenticated command relay. Run on the SSH host as an
UNPRIVILEGED user, bound to localhost, and expose it through an HTTPS tunnel.

  export RELAY_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')
  python3 server.py            # listens on 127.0.0.1:8787

Endpoints (all require header "Authorization: Bearer $RELAY_TOKEN"):
  POST /exec      {"cmd": "...", "cwd": "...", "timeout": 120}
  POST /put       {"path": "...", "b64": "...", "mode": 420}
  POST /get       {"path": "..."}   -> {"b64": "..."}
"""
import base64, hmac, json, os, subprocess, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("RELAY_TOKEN", "")
HOST = os.environ.get("RELAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("RELAY_PORT", "8787"))
ROOT = os.path.realpath(os.environ.get("RELAY_ROOT", os.path.expanduser("~")))
MAX_BODY = 64 * 1024 * 1024
MAX_OUT = 2 * 1024 * 1024
LOG = os.environ.get("RELAY_LOG", os.path.join(ROOT, ".relay.log"))

if len(TOKEN) < 32:
    sys.exit("RELAY_TOKEN must be set and at least 32 chars")


def safe(path):
    p = os.path.realpath(os.path.join(ROOT, os.path.expanduser(path)))
    if p != ROOT and not p.startswith(ROOT + os.sep):
        raise ValueError("path outside RELAY_ROOT")
    return p


def audit(line):
    with open(LOG, "a") as f:
        f.write(line.replace("\n", " ")[:2000] + "\n")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        got = self.headers.get("Authorization", "")
        if not hmac.compare_digest(got.encode(), ("Bearer " + TOKEN).encode()):
            return self.reply(401, {"error": "unauthorized"})
        n = int(self.headers.get("Content-Length", "0"))
        if n > MAX_BODY:
            return self.reply(413, {"error": "too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/exec":
                cwd = safe(req.get("cwd", "."))
                audit(f"EXEC cwd={cwd} cmd={req['cmd']}")
                try:
                    r = subprocess.run(["bash", "-lc", req["cmd"]], cwd=cwd,
                                       capture_output=True,
                                       timeout=min(int(req.get("timeout", 120)), 3600))
                    return self.reply(200, {"rc": r.returncode,
                        "stdout": r.stdout[-MAX_OUT:].decode(errors="replace"),
                        "stderr": r.stderr[-MAX_OUT:].decode(errors="replace")})
                except subprocess.TimeoutExpired:
                    return self.reply(200, {"rc": 124, "stdout": "", "stderr": "timeout"})
            if self.path == "/put":
                p = safe(req["path"])
                audit(f"PUT {p}")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(base64.b64decode(req["b64"]))
                os.chmod(p, int(req.get("mode", 0o644)))
                return self.reply(200, {"ok": True})
            if self.path == "/get":
                p = safe(req["path"])
                audit(f"GET {p}")
                with open(p, "rb") as f:
                    return self.reply(200, {"b64": base64.b64encode(f.read()).decode()})
            return self.reply(404, {"error": "not found"})
        except Exception as e:
            return self.reply(400, {"error": str(e)})


if __name__ == "__main__":
    print(f"relay on {HOST}:{PORT}, root={ROOT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
