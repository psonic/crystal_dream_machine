"""
Motore della preview: un thread che possiede Renderer, scie e posizione, e produce
i frame in tempo reale. Usa le stesse primitive del render (Renderer.step), quindi
quello che vedi è quello che esce nel video.

Pezzi:
  MaskSource   pool di processi PERSISTENTE per le deformazioni. Quando cambi un
               parametro non si riavvia: il nuovo deformatore viene scritto su un
               file (versionato) e ogni processo lo ricarica alla prima richiesta.
  BgSource     sfondo alla risoluzione della preview: un thread lo decodifica tutto
               in memoria (compresso senza perdita) mentre guardi, così saltare
               in un punto qualsiasi è istantaneo.
  Engine       il thread di riproduzione. Comandi via coda (play, seek, modifica...)

Cosa succede quando cambi un parametro:
  • solo composizione (scie, fusione, glow...)  → aggiornamento al volo
  • il resto → nuovo Renderer (~50 ms: logo/audio/probe sono in cache)
  In entrambi i casi lo STATO DELLE SCIE resta: il video continua senza scatti e,
  da fermo, il frame corrente viene ricomposto subito coi parametri nuovi.
"""

from __future__ import annotations

import math
import os
import pickle
import queue
import signal
import tempfile
import threading
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np

from .. import media
from ..config import ROOT_DIR, ConfigError, deep_merge, resolve, to_ns
from ..render import LIGHT_KEYS, Renderer
from ..procs import kill_pool, watch_parent
from .presetfile import flatten

CACHE_DIR = ROOT_DIR / ".cache" / "studio"
_MISSING = object()  # "sfondo di questo frame non ancora letto" (None = sfondo nero)
BG_BUDGET = 2_000_000_000  # byte di RAM per lo sfondo in cache


# ═════════════════════════════════════════════════════════════════════════════
#  Deformazioni: pool di processi persistente
# ═════════════════════════════════════════════════════════════════════════════
_WS: dict = {"path": None}


def _w_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ctrl-c lo gestisce il processo principale
    watch_parent()  # se lo studio muore (kill, crash, terminale chiuso) il worker esce
    cv2.setNumThreads(1)


def _w_deform(path, k):
    if _WS["path"] != path:
        with open(path, "rb") as f:
            _WS["d"], _WS["m"] = pickle.load(f)
        _WS["path"] = path
    return _WS["d"](_WS["m"], k)


class MaskSource:
    def __init__(self, workers: int):
        self.workers = workers
        self.ex = ProcessPoolExecutor(workers, initializer=_w_init)
        self.futs: dict[int, object] = {}
        self.path = None
        self.r = None
        self._files: list[str] = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for f in CACHE_DIR.glob("deform_*.pkl"):  # avanzi di sessioni precedenti
            f.unlink(missing_ok=True)

    def set(self, r: Renderer):
        self.r = r
        for f in self.futs.values():
            f.cancel()
        self.futs = {}
        if not r.deformer.active:
            self.path = None
            return
        fd, path = tempfile.mkstemp(".pkl", "deform_", str(CACHE_DIR))
        with os.fdopen(fd, "wb") as f:
            pickle.dump((r.deformer, np.asarray(r.mask0)), f, protocol=pickle.HIGHEST_PROTOCOL)
        self.path = path
        self._files.append(path)
        while len(self._files) > 4:  # le versioni vecchie non servono più
            old = self._files.pop(0)
            try:
                os.unlink(old)
            except OSError:
                pass

    def request(self, idxs):
        if self.path is None:
            return
        for i in idxs:
            if i not in self.futs:
                self.futs[i] = self.ex.submit(_w_deform, self.path, int(self.r.tl.ref_index[i]))

    def get(self, i):
        if self.path is None:
            return self.r.mask0
        self.request([i])
        return self.futs.pop(i).result()

    def drop_outside(self, lo, hi):
        for i in [i for i in self.futs if not lo <= i < hi]:
            self.futs.pop(i).cancel()

    def close(self):
        kill_pool(self.ex)
        for f in self._files:
            try:
                os.unlink(f)
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════════════
#  Sfondo: cache in memoria + lettura sequenziale
# ═════════════════════════════════════════════════════════════════════════════
def bg_key(r: Renderer):
    b = r.cfg.background
    return (r.bg_file, r.W, r.H, r.fps, r.n_frames, round(r.bg_start, 4), b.fit, b.zoom, b.slowdown,
            b.crop_x, b.crop_y, b.crop_w, b.crop_h) if r.bg_file else None


class BgSource:
    """Frame di sfondo (non ancora scuriti) per indice di output."""

    def __init__(self, r: Renderer):
        self.key = bg_key(r)
        self.file, self.W, self.H, self.fps, self.n = r.bg_file, r.W, r.H, r.fps, r.n_frames
        self.bcfg, self.t0 = r.cfg.background, r.bg_time(0)
        self.slow = r.cfg.background.slowdown
        self.frames: list[bytes | None] = [None] * self.n
        self.size = 0
        self.seq = None       # lettore sequenziale per riproduzione/riscaldamento
        self.seq_pos = -1
        self._stop = False
        self.filled = 0
        self.t = threading.Thread(target=self._fill, daemon=True, name="bg-fill")
        self.t.start()

    def _time(self, i):
        return self.t0 + i / self.fps / self.slow

    def _store(self, i, frame):
        if self.frames[i] is None and self.size < BG_BUDGET:
            b = zlib.compress(frame.tobytes(), 1)  # senza perdita: le scie dello sfondo usano i bordi
            self.frames[i] = b
            self.size += len(b)

    def _fill(self):
        """Decodifica tutto lo sfondo in memoria, dall'inizio, finché c'è budget."""
        reader = media.BackgroundReader(self.file, self.W, self.H, self.fps, self._time(0), self.bcfg, count=self.n)
        try:
            for i in range(self.n):
                if self._stop or self.size >= BG_BUDGET:
                    break
                f = reader.read()
                if f is None:
                    break
                self._store(i, f)
                self.filled = i + 1
        finally:
            reader.close()

    def cached_fraction(self):
        return sum(1 for f in self.frames if f is not None) / max(1, self.n)

    def get(self, i):
        b = self.frames[i]
        if b is not None:
            return np.frombuffer(zlib.decompress(b), np.uint8).reshape(self.H, self.W, 3)
        # il riempimento sta per arrivarci: aspetta un attimo invece di aprire un altro ffmpeg
        if self.t.is_alive() and 0 <= i - self.filled < self.fps:
            deadline = time.perf_counter() + 1.5
            while self.frames[i] is None and self.t.is_alive() and time.perf_counter() < deadline:
                time.sleep(0.005)
            if self.frames[i] is not None:
                return self.get(i)
        if self.seq is None or self.seq_pos != i:
            if self.seq:
                self.seq.close()
            self.seq = media.BackgroundReader(self.file, self.W, self.H, self.fps, self._time(i), self.bcfg,
                                              count=self.n - i)
            self.seq_pos = i
        f = self.seq.read()
        self.seq_pos += 1
        if f is not None:
            self._store(i, f)
        return f

    def close(self):
        self._stop = True
        if self.seq:
            self.seq.close()


# ═════════════════════════════════════════════════════════════════════════════
#  Il motore
# ═════════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, user: dict, name: str, on_event, workers: int | None = None):
        self.user, self.name = user, name
        self.emit = on_event                     # on_event(tipo, dati) → SSE
        self.view = {"scale": 0.5, "fps": None}  # fps None = come l'output
        self.seed = 0                            # seed di sessione (se il preset ha seed = 0)
        self.cmds: queue.Queue = queue.Queue()
        self.lock = threading.RLock()            # protegge user/effective/r letti dal server

        workers = workers or max(2, min(8, (os.cpu_count() or 4) - 2))
        self.masks = MaskSource(workers)
        self.bg: BgSource | None = None
        self.r: Renderer | None = None
        self.effective: dict = {}
        self.error: str | None = None
        self.version = 0

        self.playing = False
        self.i = 0               # prossimo frame da calcolare
        self.trails = None       # stato delle scie prima del frame i
        self.cur = None          # (indice, sfondo grezzo, scie prima) del frame mostrato
        self.cold = True         # scie non scaldate (dopo un salto senza riscaldamento)
        self.warming = None      # (fatti, totale) durante il riscaldamento

        self.frame_cond = threading.Condition()
        self._enc = ThreadPoolExecutor(1, thread_name_prefix="jpeg")
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.shown = 0
        self.stats = {"fps": 0.0, "mask": 0.0, "step": 0.0, "bg": 0.0}
        self._t_next = None
        self._last_frame_t = None
        self._last_status = 0.0

        self._rebuild(initial=True)
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self.thread.start()

    # ─── configurazione ──────────────────────────────────────────────────────
    def preview_data(self, user=None):
        """Config completo della preview: quello del preset, con seed/fps/scala della preview."""
        data = resolve(self.user if user is None else user)
        if not data["seed"]:
            if not self.seed:
                self.seed = int(np.random.default_rng().integers(1, 2**31))
            data["seed"] = self.seed
        return data

    def render_config(self) -> dict:
        """Config per il render finale: identico alla preview (stesso seed), a piena risoluzione."""
        with self.lock:
            return self.preview_data()

    def _view_cfg(self, data):
        d = deep_merge(data, {"output": {"scale": self.view["scale"]}})
        if self.view["fps"]:
            d["output"]["fps"] = self.view["fps"]
        return d

    def _rebuild(self, initial=False):
        with self.lock:
            try:
                data = self.preview_data()
                vdata = self._view_cfg(data)
                cfg = to_ns(vdata)
            except (ConfigError, ValueError) as e:
                self.error = str(e)
                self._emit_config()
                return
            old = self.r
            t_pos = (self.cur[0] if self.cur else 0) / old.fps if old else 0.0
            light = False
            if old is not None and not initial:
                a, b = flatten(old.cfg_dict), flatten(vdata)
                changed = {f"{s}.{k}" for s in set(a) | set(b) for k in set(a.get(s, {})) | set(b.get(s, {}))
                           if a.get(s, {}).get(k) != b.get(s, {}).get(k)}
                light = all(c.startswith(LIGHT_KEYS) for c in changed)
                if not changed:
                    self.effective = data
                    self.error = None
                    self._emit_config()
                    return
            try:
                if light:
                    old.update_light(cfg, vdata)
                    r = old
                else:
                    r = Renderer(cfg, vdata, self.name, log=lambda m: self.emit("toast", {"msg": m, "kind": "warn"}))
            except Exception as e:  # noqa: BLE001 — un file mancante o un valore assurdo non deve fermare lo studio
                self.error = f"{type(e).__name__}: {e}"
                if not isinstance(e, (OSError, ValueError, RuntimeError, ConfigError)):
                    traceback.print_exc()
                self._emit_config()
                return

            self.r, self.effective, self.error = r, data, None
            self.version += 1
            bg_changed = False
            if not light:
                self.masks.set(r)
                key = bg_key(r)
                old_key = self.bg.key if self.bg else None
                if key != old_key:
                    bg_changed = True
                    if self.bg:
                        self.bg.close()
                    self.bg = BgSource(r) if key else None
            reshape = old is None or (old.W, old.H, old.fps, old.n_frames) != (r.W, r.H, r.fps, r.n_frames)
            if reshape or bg_changed:
                # il frame mostrato e le scie vengono dal vecchio sfondo (o da un'altra risoluzione):
                # va riletto e riscaldato, non solo ricomposto
                i = min(r.n_frames - 1, int(round(t_pos * r.fps))) if reshape else self.cur[0]
                self.cur = (i, _MISSING, None)
                self.trails = None
            self._emit_config()
        return reshape or bg_changed

    def _emit_config(self):
        self.emit("config", self.config_payload())

    def config_payload(self):
        with self.lock:
            r = self.r
            out = {"user": self.user, "effective": self.effective, "error": self.error, "preset": self.name,
                   "seed": self.effective.get("seed", self.seed), "seed_fixed": bool(self.user.get("seed")),
                   "view": self.view, "version": self.version}
            if r:
                out.update(n=r.n_frames, fps=r.fps, duration=r.duration, W=r.W, H=r.H,
                           full=[int(round(r.W / r.s)), int(round(r.H / r.s))],
                           audio=list(r.audio) if r.audio else None,
                           background=r.bg_file, bg_start=r.bg_start)
            return out

    # ─── comandi (dal server) ────────────────────────────────────────────────
    def post(self, *cmd):
        self.cmds.put(cmd)

    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                self.playing = False
                self.emit("toast", {"msg": "errore nel motore della preview (vedi terminale)", "kind": "error"})
                time.sleep(0.2)

    def _drain(self, timeout):
        cmds = []
        try:
            cmds.append(self.cmds.get(timeout=timeout) if timeout else self.cmds.get_nowait())
            while True:
                cmds.append(self.cmds.get_nowait())
        except queue.Empty:
            pass
        return cmds

    def _tick(self):
        if self.playing:
            wait = 0.0 if self._t_next is None else max(0.0, self._t_next - time.perf_counter())
            cmds = self._drain(wait if wait > 0 else None)
        else:
            cmds = self._drain(0.25)
        if self.r is None:
            if any(c[0] == "config" for c in cmds):
                self._rebuild()
            return

        rebuild, seek, steps = False, None, 0
        for c in cmds:
            op = c[0]
            if op == "config":
                rebuild = True
            elif op == "view":
                self.view.update(c[1])
                rebuild = True
            elif op == "seek":
                seek = (c[1], c[2])
                steps = 0
            elif op == "play":
                self.playing = c[1]
                self._t_next = None
                if self.playing and self.cur and self.cur[0] >= self.r.n_frames - 1:
                    seek = (0, False)
            elif op == "toggle":
                self.playing = not self.playing
                self._t_next = None
            elif op == "step":
                self.playing = False
                steps += c[1]

        if rebuild:
            reshaped = self._rebuild()
            if reshaped and seek is None and self.cur:
                seek = (self.cur[0], True)
            elif seek is None and not self.playing:
                self._recompose()
        if seek is not None:
            self._seek(*seek)
        if steps > 0:
            for _ in range(steps):
                self._advance_one()
        elif steps < 0 and self.cur:
            self._seek(self.cur[0] + steps, True)

        if self.playing and not self._pending_seek():
            self._play_one()
        elif time.perf_counter() - self._last_status > 0.5:
            self._emit_status()

    def _pending_seek(self):
        return not self.cmds.empty()

    # ─── frame ───────────────────────────────────────────────────────────────
    def _bg(self, i):
        return self.bg.get(i) if self.bg else None

    def _produce(self, i, trails):
        """Calcola il frame i dalle scie `trails` → mostra, aggiorna stato."""
        r = self.r
        t0 = time.perf_counter()
        bg_raw = self._bg(i)
        t1 = time.perf_counter()
        self.masks.request(range(i + 1, min(r.n_frames, i + 1 + self.masks.workers * 2)))
        mask = self.masks.get(i)
        t2 = time.perf_counter()
        frame, post = r.step(i, bg_raw, mask, trails)
        t3 = time.perf_counter()
        a = 0.15
        self.stats["bg"] += a * ((t1 - t0) * 1000 - self.stats["bg"])
        self.stats["mask"] += a * ((t2 - t1) * 1000 - self.stats["mask"])
        self.stats["step"] += a * ((t3 - t2) * 1000 - self.stats["step"])
        self.cur = (i, bg_raw, trails)
        self.trails = post
        self.i = i + 1
        self._publish(frame, i)

    def _advance_one(self):
        r = self.r
        i = self.i
        if i >= r.n_frames:  # fine del video: ricomincia (le scie partono vuote, come nel video)
            i, self.trails, self.cold = 0, None, False
        self._produce(i, self.trails)

    def _play_one(self):
        now = time.perf_counter()
        fps = self.r.fps
        if self._t_next is None or now - self._t_next > 0.25:
            self._t_next = now  # troppo indietro: non rincorrere, riparti da adesso
        self._advance_one()
        self._t_next += 1 / fps

    def _recompose(self):
        """Da fermo: ricalcola il frame mostrato coi parametri nuovi (scie comprese)."""
        if not self.cur:
            return
        i, bg_raw, trails = self.cur
        if bg_raw is _MISSING:
            return self._seek(i, True)
        self.masks.drop_outside(i, i + 1)
        mask = self.masks.get(i)
        frame, post = self.r.step(i, bg_raw, mask, trails)
        self.trails, self.i = post, i + 1
        self._publish(frame, i)

    def warmup_frames(self):
        c = self.r.cfg.tracers
        n = max(c.logo.trail if c.logo.enabled else 0, c.background.trail if c.background.enabled else 0)
        return int(math.ceil(1.5 * n * self.r.fps / 30))  # la scia scende sotto il 5%

    def _seek(self, i, warm):
        r = self.r
        i = int(min(max(0, i), r.n_frames - 1))
        self._t_next = None
        self.masks.drop_outside(i, i + 1)
        self._produce(i, None)  # subito, senza scie: risposta immediata
        self.cold = i > 0
        if not warm or i == 0 or self.warmup_frames() == 0:
            return
        start = max(0, i - self.warmup_frames())
        self.masks.drop_outside(start, i + 1)
        self.masks.request(range(start, i + 1))
        trails = None
        total = i - start
        for j in range(start, i):
            if self._pending_seek():  # un nuovo comando (es. trascini ancora): lascia perdere
                self.warming = None
                return
            self.warming = (j - start, total)
            if (j - start) % 10 == 0:
                self._emit_status()
            trails = r.advance(j, r.darken(self._bg(j)), self.masks.get(j), trails)
        self.warming = None
        self.cold = False
        self._produce(i, trails)

    # ─── uscita ──────────────────────────────────────────────────────────────
    def _publish(self, frame, i):
        """Codifica JPEG su un altro thread: il ciclo di riproduzione intanto va avanti."""
        now = time.perf_counter()
        if self._last_frame_t is not None and self.playing:
            dt = now - self._last_frame_t
            if dt > 0:
                self.stats["fps"] += 0.1 * (1 / dt - self.stats["fps"])
        self._last_frame_t = now
        self._enc.submit(self._encode, frame, i)
        if not self.playing or now - self._last_status > 1 / 20:
            self._emit_status(shown=i)

    def _encode(self, frame, i):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with self.frame_cond:
            self.jpeg = buf.tobytes()
            self.frame_id += 1
            self.shown = i
            self.frame_cond.notify_all()

    def status_payload(self, shown=None):
        r = self.r
        return {
            "i": self.shown if shown is None else shown, "n": r.n_frames if r else 0, "fps": r.fps if r else 0,
            "playing": self.playing, "fps_actual": round(self.stats["fps"], 1) if self.playing else None,
            "ms": {k: round(v, 1) for k, v in self.stats.items() if k != "fps"},
            "cold": self.cold, "warming": self.warming,
            "cache": round(self.bg.cached_fraction(), 3) if self.bg else 1.0,
        }

    def _emit_status(self, shown=None):
        self._last_status = time.perf_counter()
        self.emit("status", self.status_payload(shown))

    def wait_frame(self, last_id, timeout=1.0):
        with self.frame_cond:
            if self.frame_id == last_id:
                self.frame_cond.wait(timeout)
            return self.frame_id, self.jpeg

    def close(self):
        self.masks.close()
        if self.bg:
            self.bg.close()

