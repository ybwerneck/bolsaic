# SSH-host relay (HTTPS command bridge)

Lets a cloud Claude session (HTTPS-only egress) run commands on a host it
cannot SSH to. Stdlib Python 3 only.

## On the SSH host
1. `python3 -c 'import secrets;print(secrets.token_urlsafe(48))'` -> your token.
2. `export RELAY_TOKEN=<token> RELAY_ROOT=$HOME/work; python3 server.py`
   (use tmux/screen/systemd --user so it survives logout). Run as a normal user, never root.
3. Expose 127.0.0.1:8787 over HTTPS, e.g. Cloudflare Tunnel:
   `cloudflared tunnel --url http://127.0.0.1:8787` (random *.trycloudflare.com URL;
   a named tunnel on your own domain gives a stable URL).

## In the Claude cloud environment
- Edit environment -> Network access: allow the tunnel hostname.
- Environment secrets: set `RELAY_URL` and `RELAY_TOKEN`.

## Use
    ./relay.py exec "hostname && nproc" --cwd project
    ./relay.py put local.py project/local.py
    ./relay.py get project/out.csv out.csv

Security: token is compared in constant time; commands/paths are confined to
RELAY_ROOT for cwd/put/get (note `exec` is still arbitrary shell as that user,
so treat the token like an SSH key). Every request is logged to `$RELAY_ROOT/.relay.log`.
Rotate the token by restarting the server with a new one. Stop the tunnel when not working.
