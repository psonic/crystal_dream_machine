"""
Renderer: prepara tutto (logo, sfondo, audio, timeline) e produce i frame.

Pipeline di un frame
    sfondo (ffmpeg) → scurisci → scie sfondo + scie logo → texture sfondo
    → glow → logo deformato (organic → stretch → lenti) fuso con lo sfondo
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import media
from .config import FORMATS
from .effects.compose import Trail, apply_texture, blend_logo, mask_roi, odd
from .effects.deform import Deformer, worker_deform, worker_init
from .timeline import REF_FPS, Timeline


def even(x):
    return max(2, int(round(x / 2)) * 2)


class Renderer:
    def __init__(self, cfg, cfg_dict: dict, preset_name: str = "default", log=print):
        self.cfg, self.cfg_dict, self.preset_name, self.log = cfg, cfg_dict, preset_name, log
        o = cfg.output

        # ─── dimensioni ──────────────────────────────────────────────────────
        if o.format == "custom":
            W0, H0 = o.width, o.height
        elif o.format in ("logo", "svg"):
            sw, sh = media.logo_size(cfg.logo.file)
            W0, H0 = sw + 2 * cfg.logo.padding, sh + 2 * cfg.logo.padding
        elif o.format in FORMATS:
            W0, H0 = FORMATS[o.format]
        else:
            raise ValueError(f"output.format '{o.format}' sconosciuto")
        self.W, self.H = even(W0 * o.scale), even(H0 * o.scale)
        self.s = self.W / W0
        self.fps, self.duration = o.fps, o.duration

        self.seed = int(cfg.seed) or int(np.random.default_rng().integers(1, 2**31))
        rng = np.random.default_rng(self.seed)
        self.run_info = {"seed": self.seed, "size": [self.W, self.H]}

        # ─── logo ────────────────────────────────────────────────────────────
        lg = cfg.logo
        self.mask0 = media.load_logo_mask(lg.file, self.W, self.H, lg.padding * self.s,
                                          lg.left_padding * self.s, lg.zoom, lg.fill, self.s)
        self.logo_bgr = np.array(lg.color[::-1], np.float32) / 255.0

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

        # ─── texture ─────────────────────────────────────────────────────────
        self.texture = None
        t = cfg.texture
        if t.enabled:
            img = cv2.imread(t.path, cv2.IMREAD_COLOR)
            if img is None:
                log(f"⚠️  Texture non trovata: {t.path}")
            else:
                self.texture = cv2.resize(img, (self.W, self.H)).astype(np.float32) / 255.0

        self.tl = Timeline(cfg, self.W, self.H, self.s, self.fps, self.duration, rng, bands)
        self.deformer = Deformer(cfg, self.tl, self.W, self.H, self.s, self.seed)
        self.n_frames = self.tl.n_out
        self.timing = defaultdict(float)

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

    # ═════════════════════════════════════════════════════════════════════════
    def _workers(self):
        w = self.cfg.output.workers
        if w <= 0:
            w = max(1, min(8, (os.cpu_count() or 2) - 2))
        return w if self.deformer.active else 1

    def _mask_stream(self, start, stop):
        """Maschere deformate in ordine; in parallelo se workers > 1."""
        ks = [int(self.tl.ref_index[i]) for i in range(start, stop)]
        workers = self._workers()
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
        c, tl, T = self.cfg, self.tl, self.timing

        bg_reader = None
        if self.bg_file:
            t0 = self.bg_start + first / self.fps / c.background.slowdown
            bg_reader = media.BackgroundReader(self.bg_file, self.W, self.H, self.fps, t0, c.background)
        logo_trail = Trail(c.tracers.logo, self.fps) if c.tracers.logo.enabled else None
        bg_trail = Trail(c.tracers.background, self.fps) if c.tracers.background.enabled else None
        gain = c.background.darken * c.background.contrast
        black = np.zeros((self.H, self.W, 3), np.uint8)
        workers = self._workers()
        self.log(f"  Processi : {workers}")

        try:
            masks = self._mask_stream(first, stop)
            for i in range(first, stop):
                k = int(tl.ref_index[i])
                t = time.perf_counter()

                bg = bg_reader.read() if bg_reader else None
                bg = black if bg is None else bg
                if gain != 1.0:
                    bg = cv2.convertScaleAbs(bg, alpha=gain)
                T["sfondo"] += time.perf_counter() - t; t = time.perf_counter()

                mask = next(masks)
                T["deformazioni" + (" (attesa)" if workers > 1 else "")] += time.perf_counter() - t
                t = time.perf_counter()

                frame = bg.astype(np.float32) * (1 / 255)
                if logo_trail and logo_trail.buf is not None:
                    frame += logo_trail.buf
                if bg_trail and bg_trail.buf is not None:
                    frame += bg_trail.buf
                if bg_trail:
                    g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (3, 3), 0)
                    bg_trail.push(cv2.Canny(g, c.tracers.background.threshold1, c.tracers.background.threshold2),
                                  k, tl.bg_tracer_mult[k])
                if logo_trail:
                    kd = max(2, int(round(2 * self.s)))
                    edges = cv2.dilate(cv2.Canny(mask, 50, 150), np.ones((kd, kd), np.uint8))
                    logo_trail.push(edges, k, tl.tracer_mult[k])
                np.clip(frame, 0, 1, out=frame)
                T["scie"] += time.perf_counter() - t; t = time.perf_counter()

                frame = self._composite(frame, mask, k)
                T["composizione"] += time.perf_counter() - t

                np.maximum(frame, 0, out=frame)
                yield i, cv2.convertScaleAbs(frame, alpha=255)  # satura a 0-255 e arrotonda
        finally:
            if bg_reader:
                bg_reader.close()

    def _composite(self, frame, mask, k):
        c = self.cfg
        tex = self.texture
        if tex is not None and c.texture.target in ("background", "both"):
            frame = apply_texture(frame, tex, c.texture.background_alpha, c.texture.mode)

        logo_img = None
        if tex is not None and c.texture.target in ("logo", "both"):
            inside = (mask > 0).astype(np.float32)
            logo_img = apply_texture(inside[..., None] * self.logo_bgr, tex, c.texture.alpha, c.texture.mode, inside)
            logo_img *= inside[..., None]

        if c.glow.enabled:
            ks = odd(c.glow.size * self.s)
            roi = mask_roi(mask, ks)
            if roi:
                y0, y1, x0, x1 = roi
                glow = cv2.GaussianBlur(mask[y0:y1, x0:x1], (ks, ks), 0).astype(np.float32)
                glow *= float(self.tl.glow[k]) / 255
                sub = frame[y0:y1, x0:x1]
                sub += glow[..., None] * self.logo_bgr
                np.minimum(sub, 1.0, out=sub)

        if c.blend.enabled:
            return blend_logo(frame, mask, self.logo_bgr, c.blend, self.s, logo_img)
        roi = mask_roi(mask, 1)
        if roi:
            y0, y1, x0, x1 = roi
            m3 = (mask[y0:y1, x0:x1].astype(np.float32) * (1 / 255))[..., None]
            logo = logo_img[y0:y1, x0:x1] if logo_img is not None else self.logo_bgr
            frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - m3) + logo * m3
        return frame

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

    def render_video(self, out=None) -> Path:
        path = self.output_path(out)
        o = self.cfg.output
        audio = self.audio if (self.audio and self.cfg.audio.mux) else None
        writer = media.VideoWriter(str(path), self.W, self.H, self.fps, o.crf, o.x264_preset, audio)

        # config completo accanto al video: per rifarlo identico → python -m crystal render <file>.json
        meta = {"config": self.resolved_config(), "run": self.run_info,
                "preset": self.preset_name, "created": datetime.now().isoformat(timespec="seconds")}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        t_start = time.perf_counter()
        try:
            for i, frame in self.frames():
                writer.write(frame)
                progress(i + 1, self.n_frames, time.perf_counter() - t_start)
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
        return f"   tempo medio per frame: {parts}"


def progress(done, total, elapsed):
    fps = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / fps if fps > 0 else 0
    bar = int(30 * done / total)
    print(f"\r  [{'█' * bar}{'░' * (30 - bar)}] {100 * done / total:5.1f}%  {done}/{total}  "
          f"{fps:4.1f} fps  ETA {int(eta // 60):02d}:{int(eta % 60):02d}  ", end="", flush=True)
