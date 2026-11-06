"""
Server dello studio: solo libreria standard, in ascolto su 127.0.0.1.

    GET  /                 la pagina (static/)
    GET  /stream.mjpg      il video della preview (MJPEG)
    GET  /events           eventi (SSE): config, status, render, toast
    GET  /snapshot.jpg     il frame mostrato adesso
    GET  /audio            la traccia audio della preview (il pezzo usato)
    GET  /file?path=...    un file del progetto (video renderizzati, PNG)
    GET  /api/state|schema|bands|files
    POST /api/<comando>    set, reset, user, load, save, seed, play, toggle, seek,
                           step, view, render, render_cancel, frame, open
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..config import PRESETS_DIR, ROOT_DIR, ConfigError, list_presets, load_defaults, load_user, resolve
from .engine import CACHE_DIR, Engine
from .presetfile import write_preset
from .schema import build_schema

STATIC = Path(__file__).parent / "static"
INPUT = ROOT_DIR / "input"
FILE_KINDS = {
    "logo": (".svg", ".pdf"),
    "video": (".mov", ".mp4", ".m4v", ".mkv", ".avi"),
    "audio": (".aif", ".aiff", ".wav", ".mp3", ".m4a", ".flac"),
    "image": (".jpg", ".jpeg", ".png", ".webp"),
}
SERVABLE = {".mp4", ".mov", ".png", ".jpg", ".jpeg", ".m4a", ".webp"}


# ─── accesso alle chiavi "a.b.c" ─────────────────────────────────────────────
def get_path(d, key, default=None):
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def set_path(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def del_path(d, key):
    parts = key.split(".")
    stack = []
    for p in parts[:-1]:
        if p not in d:
            return
        stack.append((d, p))
        d = d[p]
    d.pop(parts[-1], None)
    for parent, p in reversed(stack):  # via le sezioni rimaste vuote
        if not parent[p]:
            del parent[p]


def coerce(default, value):
    """Il valore dal browser, con il tipo del default (es. 3.0 → 3 per le chiavi intere)."""
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


class Studio:
    def __init__(self, preset: str | None, overrides=(), seed: int | None = None):
        self.defaults = load_defaults()
        user, name, path = load_user(preset, overrides)
        if seed is not None:
            user["seed"] = seed
        resolve(user)  # errori nel preset: subito, nel terminale
        self.path: Path | None = path
        self.saved_user = copy.deepcopy(user) if not overrides else {}
        self.clients: set[queue.Queue] = set()
        self.clients_lock = threading.Lock()
        self.render_job: RenderJob | None = None
        self.engine = Engine(user, name, self.emit)
        self.engine.post("seek", 0, False)
        self._mtime = self._file_mtime()
        threading.Thread(target=self._watch, daemon=True, name="watch").start()

    # ─── eventi ──────────────────────────────────────────────────────────────
    def emit(self, kind, data):
        if kind == "config":
            data = {**data, "dirty": self.dirty(data.get("user")), "saved": self.saved_user,
                    "path": self.rel(self.path), "presets": list_presets(), "editable": self.editable()}
        msg = f"event: {kind}\ndata: {json.dumps(data, default=_json_default)}\n\n".encode()
        with self.clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def toast(self, msg, kind="info"):
        self.emit("toast", {"msg": msg, "kind": kind})

    def dirty(self, user=None):
        return (self.engine.user if user is None else user) != self.saved_user

    def editable(self):
        return bool(self.path and self.path.suffix == ".toml" and self.path.parent.resolve() == PRESETS_DIR.resolve())

    @staticmethod
    def rel(p):
        if not p:
            return None
        try:
            return str(Path(p).resolve().relative_to(ROOT_DIR))
        except ValueError:
            return str(p)

    # ─── preset sul disco ────────────────────────────────────────────────────
    def _file_mtime(self):
        try:
            return self.path.stat().st_mtime if self.path else 0
        except OSError:
            return 0

    def _watch(self):
        """Se modifichi il preset nell'editor e salvi, lo studio lo ricarica."""
        while True:
            time.sleep(0.5)
            m = self._file_mtime()
            if m and m != self._mtime:
                self._mtime = m
                try:
                    user, _, _ = load_user(str(self.path))
                    resolve(user)
                except Exception as e:  # noqa: BLE001
                    self.toast(f"Il preset sul disco ha un errore: {e}", "error")
                    continue
                with self.engine.lock:
                    self.engine.user = user
                    self.saved_user = copy.deepcopy(user)
                self.engine.post("config")
                self.toast(f"🔄 {self.path.name} modificato fuori dallo studio: ricaricato")

    # ─── comandi ─────────────────────────────────────────────────────────────
    def set_user(self, user):
        resolve(user)  # valida (chiavi sconosciute, preset di blend...) prima di applicare
        with self.engine.lock:
            self.engine.user = user
        self.engine.post("config")
        return self.state_config(user)

    def state_config(self, user):
        try:
            eff = self.engine.preview_data(user)
        except (ConfigError, ValueError):
            eff = self.engine.effective
        return {"user": user, "effective": eff, "dirty": self.dirty(user)}

    def cmd_set(self, b):
        key, value = b["key"], b["value"]
        default = get_path(self.defaults, key)
        if default is None and key != "seed":
            raise ConfigError(f"Chiave sconosciuta '{key}'")
        value = coerce(default, value)
        with self.engine.lock:
            new = copy.deepcopy(self.engine.user)
        set_path(new, key, value)
        if key == "blend.preset":  # cambiando preset di fusione, le chiavi del preset precedente vanno via
            for k in list(new.get("blend", {})):
                if k not in ("preset", "enabled", "edge_softness"):
                    del new["blend"][k]
        base_user = copy.deepcopy(new)
        del_path(base_user, key)
        if key != "seed" and get_path(resolve(base_user), key) == value:
            new = base_user  # uguale a quello che sarebbe comunque: non serve scriverlo nel preset
        return self.set_user(new)

    def cmd_reset(self, b):
        with self.engine.lock:
            new = copy.deepcopy(self.engine.user)
        for key in b.get("keys") or [b["key"]]:
            del_path(new, key)
        return self.set_user(new)

    def cmd_user(self, b):
        return self.set_user(b["user"])

    def cmd_revert(self, b):
        """Torna al preset com'è salvato sul disco."""
        return self.set_user(copy.deepcopy(self.saved_user))

    def cmd_load(self, b):
        user, name, path = load_user(b["preset"])
        resolve(user)
        with self.engine.lock:
            self.engine.user, self.engine.name = user, name
            self.path = path
            self.saved_user = copy.deepcopy(user)
            self._mtime = self._file_mtime()
        self.engine.post("config")
        return {"ok": True}

    def cmd_save(self, b):
        name = (b.get("name") or "").strip()
        if name:
            if not re.fullmatch(r"[\w\-]+", name):
                raise ConfigError("Nome del preset: solo lettere, numeri, _ e -")
            path = PRESETS_DIR / f"{name}.toml"
            if path.exists() and path != self.path and not b.get("overwrite"):
                raise ConfigError(f"Esiste già un preset '{name}'")
        else:
            if not self.editable():
                raise ConfigError("Questo config non è un preset in presets/: usa «Salva come»")
            path, name = self.path, self.path.stem
        with self.engine.lock:
            user = copy.deepcopy(self.engine.user)
            write_preset(path, user, self.defaults, title=f"{name} — creato con Crystal Studio")
            self.path, self.engine.name = path, name
            self.saved_user = user
            self._mtime = self._file_mtime()
        self.engine.post("config")
        self.toast(f"💾 Salvato {self.rel(path)}")
        return {"ok": True, "path": self.rel(path)}

    def cmd_seed(self, b):
        action = b.get("action")
        rnd = int(np.random.default_rng().integers(1, 2**31))
        with self.engine.lock:
            new = copy.deepcopy(self.engine.user)
            cur = self.engine.effective.get("seed") or self.engine.seed
            if action == "random":
                if new.get("seed"):
                    new["seed"] = rnd
                else:
                    self.engine.seed = rnd
            elif action == "fix":
                new["seed"] = int(cur)
            elif action == "unfix":
                self.engine.seed = int(cur)
                new.pop("seed", None)
            elif action == "set":
                v = int(b["value"])
                if new.get("seed") or v == 0:
                    new["seed"] = v
                    if v == 0:
                        new.pop("seed")
                else:
                    self.engine.seed = v
        return self.set_user(new)

    def cmd_play(self, b):
        self.engine.post("play", bool(b.get("playing", True)))

    def cmd_toggle(self, b):
        self.engine.post("toggle")

    def cmd_seek(self, b):
        self.engine.post("seek", int(b["i"]), bool(b.get("warm", True)))

    def cmd_step(self, b):
        self.engine.post("step", int(b.get("d", 1)))

    def cmd_view(self, b):
        v = {}
        if "scale" in b:
            v["scale"] = float(min(1.0, max(0.1, b["scale"])))
        if "fps" in b:
            v["fps"] = int(b["fps"]) if b["fps"] else None
        self.engine.post("view", v)

    def cmd_render(self, b):
        if self.render_job and self.render_job.running():
            raise ConfigError("C'è già un render in corso")
        cfg = self.engine.render_config()
        self.engine.post("play", False)  # la CPU serve al render
        self.render_job = RenderJob(cfg, self.engine.name, bool(b.get("test")), self.emit, "render")
        return {"ok": True}

    def cmd_frame(self, b):
        if self.render_job and self.render_job.running():
            raise ConfigError("C'è già un render in corso")
        cfg = self.engine.render_config()
        r = self.engine.r
        t = round(self.engine.shown / r.fps, 3) if r else 0.0
        self.render_job = RenderJob(cfg, self.engine.name, False, self.emit, "frame", at=t)
        return {"ok": True}

    def cmd_render_cancel(self, b):
        if self.render_job:
            self.render_job.cancel()

    def cmd_open(self, b):
        p = safe_path(b["path"])
        subprocess.run(["open", "-R", str(p)] if b.get("reveal") else ["open", str(p)], check=False)

    # ─── letture ─────────────────────────────────────────────────────────────
    def files(self):
        out = {k: [] for k in FILE_KINDS}
        for f in sorted(INPUT.iterdir()) if INPUT.exists() else []:
            if f.name.startswith(".") or f.name.startswith("._"):
                continue
            for kind, exts in FILE_KINDS.items():
                if f.suffix.lower() in exts:
                    out[kind].append(f"input/{f.name}")
        return out

    def bands(self):
        r = self.engine.r
        if not r or r.bands is None:
            return {"bands": None}
        out = {}
        for k, v in r.bands.items():
            n = len(v)
            m = min(n, 1200)
            edges = np.linspace(0, n, m + 1).astype(int)
            out[k] = [round(float(v[a:max(a + 1, b)].max()), 3) for a, b in zip(edges[:-1], edges[1:])]
        return {"bands": out, "duration": r.duration}

    def audio_file(self):
        r = self.engine.r
        if not r or not r.audio:
            return None
        file, start, dur = r.audio
        h = hashlib.sha1(f"{Path(file).resolve()}|{Path(file).stat().st_mtime}|{start:.3f}|{dur:.3f}".encode()).hexdigest()[:16]
        out = CACHE_DIR / f"audio_{h}.m4a"
        if not out.exists():
            tmp = out.with_suffix(".tmp.m4a")
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                            "-i", file, "-vn", "-c:a", "aac", "-b:a", "192k", str(tmp)], check=True)
            tmp.replace(out)
        return out

    def state(self):
        e = self.engine
        cfg = e.config_payload()
        cfg.update(dirty=self.dirty(), saved=self.saved_user, path=self.rel(self.path), presets=list_presets(),
                   editable=self.editable())
        return {"config": cfg, "status": e.status_payload(), "files": self.files(),
                "render": self.render_job.payload() if self.render_job else None}


def safe_path(p: str) -> Path:
    path = (ROOT_DIR / p).resolve()
    if ROOT_DIR.resolve() not in path.parents:
        raise ConfigError("Percorso fuori dal progetto")
    if not path.exists():
        raise ConfigError(f"File non trovato: {p}")
    return path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Render e frame PNG in un processo a parte (`python -m crystal ...`)
# ═════════════════════════════════════════════════════════════════════════════
_PROG = re.compile(r"(\d+)/(\d+)\s+([\d.]+) fps\s+ETA (\d+):(\d+)")


class RenderJob:
    def __init__(self, cfg: dict, name: str, test: bool, emit, kind: str, at: float = 0.0):
        self.kind, self.test, self.emit = kind, test, emit
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_file = CACHE_DIR / f"{kind}_{stamp}.json"
        cfg_file.write_text(json.dumps({"config": cfg, "preset": name}, ensure_ascii=False, default=_json_default))
        if kind == "render":
            cmd = [sys.executable, "-u", "-m", "crystal", "render", str(cfg_file)] + (["--test"] if test else [])
        else:
            out = Path(cfg["output"]["dir"]) / f"frame_{name}_{at:g}s.png"
            cmd = [sys.executable, "-u", "-m", "crystal", "frame", str(cfg_file), "--at", str(at), "-o", str(out)]
        self.state = {"kind": kind, "test": test, "state": "running", "done": 0, "total": 0, "fps": 0.0,
                      "eta": "", "path": None, "log": "", "started": time.time()}
        self.proc = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._read, daemon=True, name="render").start()
        self.emit("render", self.payload())

    def running(self):
        return self.proc.poll() is None

    def payload(self):
        return dict(self.state)

    def _read(self):
        buf, lines, last_emit = b"", [], 0.0
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                line = p.decode(errors="replace").strip()
                if not line:
                    continue
                m = _PROG.search(line)
                if m:
                    d, t, fps, mm, ss = m.groups()
                    self.state.update(done=int(d), total=int(t), fps=float(fps), eta=f"{mm}:{ss}")
                else:
                    lines.append(line)
                    if line.startswith("✅"):
                        self.state["path"] = line[1:].strip().split("   ")[0].strip()
            if time.time() - last_emit > 0.25:
                last_emit = time.time()
                self.emit("render", self.payload())
        code = self.proc.wait()
        self.state["log"] = "\n".join(lines[-12:])
        if self.state["state"] != "cancelled":
            self.state["state"] = "done" if code == 0 else "error"
        self.emit("render", self.payload())

    def cancel(self):
        if self.running():
            self.state["state"] = "cancelled"
            self.proc.terminate()


# ═════════════════════════════════════════════════════════════════════════════
#  HTTP
# ═════════════════════════════════════════════════════════════════════════════
def make_handler(studio: Studio):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        # ─── risposte ────────────────────────────────────────────────────────
        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=_json_default).encode())

        def _file(self, path: Path, ctype=None):
            ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            size = path.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if a and b else size - 1
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        # ─── GET ─────────────────────────────────────────────────────────────
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    return self._file(STATIC / "index.html", "text/html; charset=utf-8")
                if u.path.startswith("/static/"):
                    p = (STATIC / u.path[len("/static/"):]).resolve()
                    if STATIC.resolve() not in p.parents or not p.exists():
                        return self._send(404, b"not found", "text/plain")
                    return self._file(p)
                if u.path == "/stream.mjpg":
                    return self._stream()
                if u.path == "/events":
                    return self._events()
                if u.path == "/snapshot.jpg":
                    _, jpeg = studio.engine.wait_frame(-1, 0)
                    return self._send(200, jpeg or b"", "image/jpeg")
                if u.path == "/audio":
                    f = studio.audio_file()
                    return self._file(f, "audio/mp4") if f else self._send(404, b"", "text/plain")
                if u.path == "/file":
                    p = safe_path(q["path"][0])
                    if p.suffix.lower() not in SERVABLE:
                        return self._send(403, b"", "text/plain")
                    return self._file(p)
                if u.path == "/api/state":
                    return self._json(studio.state())
                if u.path == "/api/schema":
                    return self._json(build_schema())
                if u.path == "/api/bands":
                    return self._json(studio.bands())
                if u.path == "/api/files":
                    return self._json(studio.files())
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ConfigError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_HEAD(self):
            self.do_GET()

        # ─── POST /api/<comando> ─────────────────────────────────────────────
        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                name = u.path.removeprefix("/api/")
                fn = getattr(studio, "cmd_" + name, None)
                if not fn or not re.fullmatch(r"\w+", name):
                    return self._json({"error": "comando sconosciuto"}, 404)
                self._json(fn(body) or {"ok": True})
            except (ConfigError, ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        # ─── flussi ──────────────────────────────────────────────────────────
        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last = -1
            try:
                # il separatore va scritto SUBITO dopo ogni frame: il browser mostra una parte
                # solo quando vede il separatore successivo (da fermo non ne arriverebbero altri)
                self.wfile.write(b"--frame\r\n")
                while True:
                    fid, jpeg = studio.engine.wait_frame(last, 1.0)
                    if fid == last or jpeg is None:
                        continue
                    last = fid
                    self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                                     + b"\r\nX-Frame: " + str(studio.engine.shown).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n--frame\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=500)
            with studio.clients_lock:
                studio.clients.add(q)
            try:
                st = studio.state()
                for kind in ("config", "status"):
                    self.wfile.write(f"event: {kind}\ndata: {json.dumps(st[kind], default=_json_default)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = b": ping\n\n"
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with studio.clients_lock:
                    studio.clients.discard(q)
            self.close_connection = True

    return H


def serve(preset=None, overrides=(), seed=None, port=8765, open_browser=True):
    studio = Studio(preset, overrides, seed)
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), make_handler(studio))
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"Nessuna porta libera fra {port} e {port + 19}")
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"\n✨ Crystal Studio → {url}")
    print(f"   preset: {studio.engine.name}   ·   ctrl-c per chiudere\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n⏹  studio chiuso")
    finally:
        httpd.server_close()
        studio.engine.close()
        if studio.render_job and studio.render_job.running():
            studio.render_job.cancel()
