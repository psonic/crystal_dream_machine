"""
Pannello di controllo: una pagina per vedere cosa gira e spegnerlo.

    GET  /                  la pagina
    GET  /api/status        processi Crystal (studio, render, worker orfani) + CPU
    GET  /api/log?pid=N     ultime righe del log di un processo avviato dal pannello
    POST /api/studio        {preset, workers, idle_exit}  avvia uno studio
    POST /api/render        {preset, test}                avvia un render
    POST /api/stop          {pid}                         ferma un processo (e i suoi worker)
    POST /api/clean                                       uccide i worker orfani
    POST /api/stop_all                                    ferma tutto (tranne il pannello)
    POST /api/open          {what: "output"}              apre la cartella nel Finder
    POST /api/quit                                        chiude il pannello

Gli studio e i render partono staccati dal pannello: se chiudi il pannello continuano
(lo studio si spegne comunque da solo quando nessuno lo guarda).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import ROOT_DIR, list_presets
from ..procs import RUN_DIR, Census, Registration, graceful_signals, kill_all, stop

STATIC = Path(__file__).parent / "static"
LOG_DIR = RUN_DIR / "logs"


def default_workers():
    return max(2, min(8, (os.cpu_count() or 4) - 2))


class Panel:
    def __init__(self):
        self.census = Census()
        self.lock = threading.Lock()
        self.children: dict[int, subprocess.Popen] = {}  # avviati da qui: vanno "raccolti" (zombie)
        self.logs: dict[int, Path] = {}
        self.stopping: set[int] = set()
        self._last = None
        self._last_t = 0.0
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(LOG_DIR.glob("*.log"))[:-30]:  # tiene solo gli ultimi 30 log
            f.unlink(missing_ok=True)

    # ─── stato ───────────────────────────────────────────────────────────────
    def status(self):
        with self.lock:
            for pid, p in list(self.children.items()):
                if p.poll() is not None:
                    del self.children[pid]
            if self._last is None or time.time() - self._last_t > 0.8:  # più pagine aperte: una lettura sola
                self._last = self.census.scan()
                self._last_t = time.time()
            s = self._last
        for m in s["processes"]:
            m["stopping"] = m["pid"] in self.stopping
            m["has_log"] = m["pid"] in self.logs or bool(m["info"].get("log"))
            if m["kind"] == "studio" and not m["info"].get("preset"):
                m["info"]["preset"] = _arg_after(m["cmd"], "studio")
            if m["kind"] == "render" and not m["info"].get("preset"):
                m["info"]["preset"] = _arg_after(m["cmd"], "render")
        return {**s, "me": os.getpid(), "presets": list_presets(),
                "default_workers": default_workers(), "now": time.time()}

    # ─── comandi ─────────────────────────────────────────────────────────────
    def _spawn(self, kind, args):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = LOG_DIR / f"{stamp}_{kind}.log"
        f = open(log, "wb")
        p = subprocess.Popen([sys.executable, "-u", "-m", "crystal", kind, *args], cwd=ROOT_DIR,
                             stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env={**os.environ, "CRYSTAL_LOG": str(log)},
                             start_new_session=True)  # sopravvive al pannello, non al suo ctrl-c
        f.close()
        with self.lock:
            self.children[p.pid] = p
            self.logs[p.pid] = log
            self._last = None
        return {"ok": True, "pid": p.pid}

    def cmd_studio(self, b):
        args = []
        if b.get("preset"):
            args.append(str(b["preset"]))
        workers = int(b.get("workers") or default_workers())
        idle = float(b.get("idle_exit", 15))
        args += ["--no-browser", "--workers", str(max(1, min(16, workers))), "--idle-exit", f"{idle:g}"]
        return self._spawn("studio", args)

    def cmd_render(self, b):
        args = [str(b["preset"])] if b.get("preset") else []
        if b.get("test"):
            args.append("--test")
        if b.get("workers"):
            args += ["--set", f"output.workers={int(b['workers'])}"]
        return self._spawn("render", args)

    def cmd_stop(self, b):
        pid = int(b["pid"])
        if pid == os.getpid():
            raise ValueError("per chiudere il pannello usa «chiudi pannello»")
        known = {m["pid"] for m in self.status()["processes"]}
        if pid not in known:
            raise ValueError("non è un processo Crystal")
        self.stopping.add(pid)

        def run():
            try:
                stop(pid)
            finally:
                self.stopping.discard(pid)
                with self.lock:
                    self._last = None
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def cmd_clean(self, _b):
        s = self.status()
        kill_all(s["orphans"])
        with self.lock:
            self._last = None
        return {"ok": True, "killed": len(s["orphans"])}

    def cmd_stop_all(self, _b):
        s = self.status()
        for m in s["processes"]:
            if m["pid"] != os.getpid():
                self.cmd_stop({"pid": m["pid"]})
        kill_all(s["orphans"])
        return {"ok": True}

    def cmd_open(self, b):
        target = {"output": ROOT_DIR / "output", "presets": ROOT_DIR / "presets"}.get(b.get("what"), ROOT_DIR)
        subprocess.Popen(["open", str(target)])
        return {"ok": True}

    def log(self, pid: int, lines=60):
        path = self.logs.get(pid)
        if not path:
            info = self.status()
            m = next((m for m in info["processes"] if m["pid"] == pid), None)
            path = Path(m["info"]["log"]) if m and m["info"].get("log") else None
        if not path or not path.exists():
            return ""
        data = path.read_bytes()[-20000:].decode(errors="replace")
        rows = [r.split("\r")[-1] for r in data.split("\n")]  # le barre di avanzamento usano \r
        return "\n".join(rows[-lines:])


def _arg_after(cmd, word):
    if word in cmd:
        i = cmd.index(word)
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
            return Path(cmd[i + 1]).stem
    return "default"


def make_handler(panel: Panel, httpd_ref):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str).encode())

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/ping":
                    return self._json({"crystal_panel": True, "pid": os.getpid()})
                if u.path == "/api/status":
                    return self._json(panel.status())
                if u.path == "/api/log":
                    pid = int(parse_qs(u.query)["pid"][0])
                    return self._json({"log": panel.log(pid)})
                self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                name = u.path.removeprefix("/api/")
                if name == "quit":
                    self._json({"ok": True})
                    threading.Thread(target=httpd_ref[0].shutdown, daemon=True).start()
                    return
                fn = getattr(panel, "cmd_" + name, None) if name.isidentifier() else None
                if not fn:
                    return self._json({"error": "comando sconosciuto"}, 404)
                self._json(fn(body) or {"ok": True})
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def _already_running(port) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1) as r:
            return bool(json.load(r).get("crystal_panel"))
    except Exception:  # noqa: BLE001
        return False


def serve(port=8700, open_browser=True):
    url = f"http://127.0.0.1:{port}/"
    if _already_running(port):  # doppio clic sull'app con il pannello già aperto: basta il browser
        print(f"Il pannello è già aperto → {url}")
        if open_browser:
            webbrowser.open(url)
        return
    graceful_signals()
    panel = Panel()
    ref = [None]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(panel, ref))
    httpd.daemon_threads = True
    ref[0] = httpd
    reg = Registration("panel", url=url, port=port)
    print(f"\n🎛  Crystal — pannello → {url}\n")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        reg.remove()
        httpd.server_close()
        print("⏹  pannello chiuso")
