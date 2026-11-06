"""
Renderer: prepara tutto (logo, sfondo, audio, timeline) e produce i frame.

Pipeline di un frame i
    sfondo (ffmpeg) → scurisci ─┬─ compose: sfondo + scie → texture → glow → logo deformato fuso
    maschera deformata (k) ─────┴─ advance: aggiorna le scie con i bordi di sfondo e logo
compose e advance leggono lo stesso stato delle scie ("prima del frame i") e sono
indipendenti: girano in parallelo su due thread. Le deformazioni girano su più processi.

La stessa logica (Renderer.step) la usano il render, `frame` e lo studio (preview web):
quello che vedi in preview è quello che esce nel video.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import media
from .config import FORMATS
from .effects.compose import Trail, add_glow, apply_texture, blend_logo, paste_logo, scalar
from .effects.deform import Deformer, worker_deform, worker_init
from .timeline import REF_FPS, Timeline, glow_curve

_THREADS = ThreadPoolExecutor(2, thread_name_prefix="crystal")
# composizioni di frame diversi in parallelo (OpenCV e numpy rilasciano il GIL)
_COMPOSE = ThreadPoolExecutor(4, thread_name_prefix="crystal-compose")


def even(x):
    return max(2, int(round(x / 2)) * 2)


def output_size(cfg) -> tuple[int, int, float]:
    """(W, H, s): dimensioni in pixel e fattore di scala rispetto alla risoluzione piena."""
    o = cfg.output
    if o.format == "custom":
        W0, H0 = o.width, o.height
    elif o.format in ("logo", "svg"):
        sw, sh = media.logo_size(cfg.logo.file)
        W0, H0 = sw + 2 * cfg.logo.padding, sh + 2 * cfg.logo.padding
    elif o.format in FORMATS:
        W0, H0 = FORMATS[o.format]
    else:
        raise ValueError(f"output.format '{o.format}' sconosciuto")
    W, H = even(W0 * o.scale), even(H0 * o.scale)
    return W, H, W / W0


# Parametri che cambiano solo la composizione del frame (non timeline, deformazioni,
# logo o sfondo): per questi lo studio aggiorna il Renderer al volo invece di rifarlo.
LIGHT_KEYS = ("tracers.", "blend.", "glow.", "texture.", "logo.color",
              "background.brightness")


class Renderer:
    def __init__(self, cfg, cfg_dict: dict, preset_name: str = "default", log=print):
        self.cfg, self.cfg_dict, self.preset_name, self.log = cfg, cfg_dict, preset_name, log
        o = cfg.output
        self.W, self.H, self.s = output_size(cfg)
        self.fps, self.duration = o.fps, o.duration

        self.seed = int(cfg.seed) or int(np.random.default_rng().integers(1, 2**31))
        rng = np.random.default_rng(self.seed)
        self.run_info = {"seed": self.seed, "size": [self.W, self.H]}

        # ─── logo ────────────────────────────────────────────────────────────
        lg = cfg.logo
        self.mask0 = media.load_logo_mask(lg.file, self.W, self.H, lg.padding * self.s,
                                          lg.left_padding * self.s, lg.zoom, lg.fill, self.s)

        # ─── sfondo ──────────────────────────────────────────────────────────
        self.bg_file, self.bg_start = None, 0.0
        b = cfg.background
        if b.enabled:
            files = media.expand_files(b.video)
            if not files:
                log(f"⚠️  Video di sfondo non trovato ({b.video}): uso sfondo nero")
            else:
                self.bg_file = files[rng.integers(len(files))]
                dur = media.probe_duration(self.bg_file)
                needed = self.duration / b.slowdown
                if b.random_start and dur > needed:
                    self.bg_start = float(rng.uniform(0, max(0.0, min(dur * 2 / 3, dur - needed))))
                elif not b.random_start:
                    self.bg_start = float(b.start)
                self.run_info.update(background=self.bg_file, background_start=round(self.bg_start, 3))

        # ─── audio ───────────────────────────────────────────────────────────
        self.audio = None
        bands = None
        a = cfg.audio
        if a.enabled:
            files = media.expand_files(a.files)
            if not files:
                log(f"⚠️  Nessun file audio trovato ({a.files}): niente reattività audio")
            else:
                file = files[rng.integers(len(files))]
                dur = media.probe_duration(file)
                if a.random_start and dur > self.duration:
                    start = float(rng.uniform(0, min(dur - self.duration, dur * 0.67)))
                else:
                    start = 0.0 if a.random_start else float(a.start)
                y = media.decode_audio(file, start, self.duration)
                bands = media.audio_bands(y, 22050, REF_FPS, int(round(self.duration * REF_FPS)))
                self.audio = (file, start, self.duration)
                self.run_info.update(audio=file, audio_start=round(start, 3))

        self.bands = bands
        self.tl = Timeline(cfg, self.W, self.H, self.s, self.fps, self.duration, rng, bands, self.mask0)
        self.deformer = Deformer(cfg, self.tl, self.W, self.H, self.s, self.seed)
        self.n_frames = self.tl.n_out
        self.timing = defaultdict(float)
        self.black = np.zeros((self.H, self.W, 3), np.uint8)
        self._setup_compose(cfg)

    def _setup_compose(self, cfg):
        """Tutto ciò che dipende solo dai LIGHT_KEYS."""
        self.logo_bgr = np.array(cfg.logo.color[::-1], np.float32) / 255.0
        self.texture = None
        t = cfg.texture
        if t.enabled:
            img = _read_texture(t.path)
            if img is None:
                self.log(f"⚠️  Texture non trovata: {t.path}")
            else:
                self.texture = cv2.resize(img, (self.W, self.H)).astype(np.float32) / 255.0
        tr = cfg.tracers
        self.logo_trail = Trail(tr.logo, self.fps) if tr.logo.enabled else None
        self.bg_trail = Trail(tr.background, self.fps) if tr.background.enabled else None
        self.gain = cfg.background.brightness
        self.tl.glow = glow_curve(cfg.glow.intensity, self.tl.n_ref)

    def update_light(self, cfg, cfg_dict):
        """Applica una modifica che riguarda solo LIGHT_KEYS, senza rifare timeline e deformazioni."""
        self.cfg, self.cfg_dict = cfg, cfg_dict
        self._setup_compose(cfg)

    # ═════════════════════════════════════════════════════════════════════════
    def describe(self):
        c = self.cfg
        on = lambda x: "✓" if x else "·"
        lines = [
            f"  Preset   : {self.preset_name}   (seed {self.seed})",
            f"  Formato  : {self.W}×{self.H} @ {self.fps}fps, {self.duration}s = {self.n_frames} frame",
            f"  Sfondo   : {self.bg_file or 'nero'}" + (f"  da {self.bg_start:.1f}s" if self.bg_file else ""),
            f"  Audio    : {self.audio[0] + f'  da {self.audio[1]:.1f}s' if self.audio else '—'}",
            f"  Effetti  : organic {on(c.organic.enabled)}  stretch {on(c.stretch.enabled)}  "
            f"lenti {on(c.lenses.enabled)}{f'({c.lenses.count})' if c.lenses.enabled else ''}  "
            f"scie {on(c.tracers.logo.enabled)}/{on(c.tracers.background.enabled)}  "
            f"glow {on(c.glow.enabled)}  blend {c.blend.preset if c.blend.enabled else '·'}",
        ]
        return "\n".join(lines)

    def bg_time(self, i: int) -> float:
        """Secondo del video di sfondo che corrisponde al frame di output i."""
        return self.bg_start + i / self.fps / self.cfg.background.slowdown

    # ═════════════════════════════════════════════════════════════════════════
    #  Un frame: le primitive usate da render, frame e studio
    # ═════════════════════════════════════════════════════════════════════════
    def mask(self, i: int) -> np.ndarray:
        return self.deformer(self.mask0, int(self.tl.ref_index[i]))

    def darken(self, bg_raw):
        bg = self.black if bg_raw is None else bg_raw
        return cv2.convertScaleAbs(bg, alpha=self.gain) if self.gain != 1.0 else bg

    def advance(self, i, bg, mask, trails):
        """Stato delle scie dopo il frame i. `trails` = (buffer logo, buffer sfondo) o None.
        Le due scie sono indipendenti: quella dello sfondo va su un altro thread."""
        t = time.perf_counter()
        k = int(self.tl.ref_index[i])
        logo_buf, bg_buf = trails or (None, None)
        fut = _THREADS.submit(self._advance_bg, k, bg, bg_buf) if self.bg_trail else None
        if self.logo_trail:
            edges, binary = self._edges(mask, _logo_edges)
            logo_buf = self.logo_trail.push(logo_buf, edges, k, binary=binary)
        if fut:
            bg_buf = fut.result()
        self.timing["scie"] += time.perf_counter() - t
        return logo_buf, bg_buf

    def _advance_bg(self, k, bg, bg_buf):
        c = self.cfg.tracers.background
        gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        # soglia bassa di Canny = ~1/5 di quella alta (il rapporto che si usava sempre)
        edges, binary = self._edges(gray, lambda g: cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0),
                                                             c.threshold * 0.22, c.threshold))
        return self.bg_trail.push(bg_buf, edges, k, binary=binary)

    def _edges(self, img, detect):
        """
        Bordi per le scie, uguali a quelli della risoluzione piena anche quando si renderizza
        più piccolo (preview, --test): si cercano su un'immagine portata alla scala piena e si
        riportano giù come copertura (quanta parte di ogni pixel è bordo). Senza, a metà
        risoluzione le scie verrebbero 2-3 volte più fitte e luminose del video finale.
        Ritorna (mappa uint8, binaria?).
        """
        if self.s >= 0.99:
            return detect(img), True
        H, W = img.shape
        up = cv2.resize(img, (int(round(W / self.s)), int(round(H / self.s))), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(detect(up), (W, H), interpolation=cv2.INTER_AREA), False

    def compose(self, i, bg, mask, trails):
        """Il frame i (BGR uint8): sfondo + scie, texture, glow, logo fuso."""
        t = time.perf_counter()
        c, k = self.cfg, int(self.tl.ref_index[i])
        frame = cv2.multiply(bg, scalar(1 / 255), dtype=cv2.CV_32F)
        for buf in trails or ():
            if buf is not None:
                cv2.add(frame, buf, dst=frame)
        cv2.threshold(frame, 1.0, 1.0, cv2.THRESH_TRUNC, dst=frame)

        tex = self.texture
        if tex is not None and c.texture.target in ("background", "both"):
            frame = np.ascontiguousarray(apply_texture(frame, tex, c.texture.background_alpha, c.texture.mode),
                                         np.float32)
        logo_img = None
        if tex is not None and c.texture.target in ("logo", "both"):
            inside = (mask > 0).astype(np.float32)
            logo_img = apply_texture(inside[..., None] * self.logo_bgr, tex, c.texture.alpha, c.texture.mode, inside)
            logo_img = np.ascontiguousarray(logo_img * inside[..., None], np.float32)

        if c.glow.enabled:
            add_glow(frame, mask, self.logo_bgr, c.glow.size * self.s, float(self.tl.glow[k]))
        if c.blend.enabled:
            blend_logo(frame, mask, self.logo_bgr, c.blend, self.s, logo_img)
        else:
            paste_logo(frame, mask, self.logo_bgr, logo_img)

        cv2.threshold(frame, 0.0, 0.0, cv2.THRESH_TOZERO, dst=frame)
        out = cv2.convertScaleAbs(frame, alpha=255)  # satura a 0-255 e arrotonda
        self.timing["composizione"] += time.perf_counter() - t
        return out

    def step(self, i, bg_raw, mask, trails, compose=True):
        """(frame i o None, scie dopo il frame i). compose e advance in parallelo."""
        bg = self.darken(bg_raw)
        if not compose:
            return None, self.advance(i, bg, mask, trails)
        fut = _COMPOSE.submit(self.compose, i, bg, mask, trails)
        trails = self.advance(i, bg, mask, trails)
        return fut.result(), trails

    # ═════════════════════════════════════════════════════════════════════════
    #  Sequenza di frame (render e `frame`)
    # ═════════════════════════════════════════════════════════════════════════
    def _workers(self):
        w = self.cfg.output.workers
        if w <= 0:
            w = max(1, min(8, (os.cpu_count() or 2) - 2))
        return w if self.deformer.active else 1

    def _mask_stream(self, start, stop):
        """Maschere deformate in ordine; in parallelo se workers > 1."""
        ks = [int(self.tl.ref_index[i]) for i in range(start, stop)]
        workers = min(self._workers(), max(1, len(ks) // 4))
        if workers <= 1:
            for k in ks:
                yield self.deformer(self.mask0, k)
            return
        with ProcessPoolExecutor(workers, initializer=worker_init,
                                 initargs=(self.deformer, self.mask0)) as ex:
            pending = deque()
            it = iter(ks)
            for k in it:
                pending.append(ex.submit(worker_deform, k))
                if len(pending) >= workers * 3:
                    break
            for k in it:
                yield pending.popleft().result()
                pending.append(ex.submit(worker_deform, k))
            while pending:
                yield pending.popleft().result()

    def frames(self, start=0, stop=None, warmup=0):
        """Genera (indice, frame BGR uint8). `warmup` frame prima di start servono a riempire le scie."""
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        first = max(0, start - warmup)
        T = self.timing

        bg_reader = None
        if self.bg_file:
            bg_reader = media.BackgroundReader(self.bg_file, self.W, self.H, self.fps, self.bg_time(first),
                                               self.cfg.background, count=stop - first)
        self.log(f"  Processi : {min(self._workers(), max(1, (stop - first) // 4))}")
        # le scie vanno calcolate in ordine (advance), ma ogni frame dipende solo dallo
        # stato delle scie prima di lui: le composizioni girano in parallelo su più thread
        trails, pending = None, deque()
        try:
            masks = self._mask_stream(first, stop)
            for i in range(first, stop):
                t = time.perf_counter()
                bg = bg_reader.read() if bg_reader else None
                T["sfondo (attesa)"] += time.perf_counter() - t
                t = time.perf_counter()
                mask = next(masks)
                T["deformazioni (attesa)"] += time.perf_counter() - t
                bg = self.darken(bg)
                if i >= start:
                    pending.append((i, _COMPOSE.submit(self.compose, i, bg, mask, trails)))
                trails = self.advance(i, bg, mask, trails)
                while pending and (len(pending) > _COMPOSE._max_workers or pending[0][1].done()):
                    j, fut = pending.popleft()
                    yield j, fut.result()
            while pending:
                j, fut = pending.popleft()
                yield j, fut.result()
        finally:
            for _, fut in pending:
                fut.cancel()
            if bg_reader:
                bg_reader.close()

    # ═════════════════════════════════════════════════════════════════════════
    def resolved_config(self) -> dict:
        """Config con le scelte casuali fissate (seed, file e punti di partenza): rifà lo stesso video."""
        d = copy.deepcopy(self.cfg_dict)
        d["seed"] = self.seed
        if self.bg_file:
            d["background"].update(video=self.bg_file, random_start=False, start=round(self.bg_start, 3))
        if self.audio:
            d["audio"].update(files=[self.audio[0]], random_start=False, start=round(self.audio[1], 3))
        return d

    def output_path(self, out=None) -> Path:
        if out:
            return Path(out)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.cfg.output.dir) / f"{stamp}_{self.preset_name}.mp4"

    def render_video(self, out=None, on_progress=None) -> Path:
        path = self.output_path(out)
        o = self.cfg.output
        audio = self.audio if (self.audio and self.cfg.audio.mux) else None
        writer = media.VideoWriter(str(path), self.W, self.H, self.fps, o.crf, o.x264_preset, audio, o.encoder)

        # config completo accanto al video: per rifarlo identico → python -m crystal render <file>.json
        meta = {"config": self.resolved_config(), "run": self.run_info,
                "preset": self.preset_name, "created": datetime.now().isoformat(timespec="seconds")}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        on_progress = on_progress or progress
        t_start = time.perf_counter()
        try:
            for i, frame in self.frames():
                writer.write(frame)
                on_progress(i + 1, self.n_frames, time.perf_counter() - t_start)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        total = time.perf_counter() - t_start
        self.log(f"\n\n✅ {path}   ({path.stat().st_size / 1e6:.1f} MB, {total:.0f}s, "
                 f"{self.n_frames / total:.1f} fps)")
        self.log(self.timing_report(total))
        return path

    def timing_report(self, total):
        parts = "  ".join(f"{k} {v / self.n_frames * 1000:.0f}ms" for k, v in self.timing.items())
        return f"   tempo medio per frame (sommato sui thread): {parts}"


_K2 = np.ones((2, 2), np.uint8)


def _logo_edges(mask):
    """Contorno del logo per la sua scia (a risoluzione piena: 2 px di spessore)."""
    return cv2.dilate(cv2.Canny(mask, 50, 150), _K2)


@media.cached(8)
def _read_texture(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def progress(done, total, elapsed):
    fps = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / fps if fps > 0 else 0
    bar = int(30 * done / total)
    print(f"\r  [{'█' * bar}{'░' * (30 - bar)}] {100 * done / total:5.1f}%  {done}/{total}  "
          f"{fps:4.1f} fps  ETA {int(eta // 60):02d}:{int(eta % 60):02d}  ", end="", flush=True)
