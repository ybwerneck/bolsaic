#!/usr/bin/env python3
"""Client. Needs RELAY_URL (https://...) and RELAY_TOKEN in the environment.
  relay.py exec "ls -la" [--cwd DIR] [--timeout S]
  relay.py put LOCAL REMOTE
  relay.py get REMOTE LOCAL
"""
import argparse, base64, json, os, sys, urllib.request

URL = os.environ["RELAY_URL"].rstrip("/")
TOKEN = os.environ["RELAY_TOKEN"]


def call(path, body, timeout):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(),
        {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="c", required=True)
e = sp.add_parser("exec"); e.add_argument("cmd"); e.add_argument("--cwd", default="."); e.add_argument("--timeout", type=int, default=120)
p = sp.add_parser("put"); p.add_argument("local"); p.add_argument("remote")
g = sp.add_parser("get"); g.add_argument("remote"); g.add_argument("local")
a = ap.parse_args()

if a.c == "exec":
    r = call("/exec", {"cmd": a.cmd, "cwd": a.cwd, "timeout": a.timeout}, a.timeout + 30)
    sys.stdout.write(r.get("stdout", "")); sys.stderr.write(r.get("stderr", ""))
    sys.exit(r.get("rc", 1))
if a.c == "put":
    d = open(a.local, "rb").read()
    mode = os.stat(a.local).st_mode & 0o777
    print(call("/put", {"path": a.remote, "b64": base64.b64encode(d).decode(), "mode": mode}, 300))
if a.c == "get":
    open(a.local, "wb").write(base64.b64decode(call("/get", {"path": a.remote}, 300)["b64"]))
