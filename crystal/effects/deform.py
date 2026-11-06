"""
Deformazioni della maschera del logo: organic → stretch → lenti.

Tutto quello che cambia nel tempo è precalcolato nella Timeline, quindi deformare
il frame k è una funzione pura (maschera, k) → maschera: si può fare in parallelo
su più processi.

Per aggiungere una deformazione nuova: scrivi un metodo `_nome(self, mask, k)` e
aggiungilo a STAGES (+ la sua sezione in defaults.toml).
"""

import math

import cv2
import numpy as np

from .noise import Perlin

NOISE_GRID = 6  # il rumore si calcola ogni N px e poi si interpola (è comunque liscio)


class Deformer:
    STAGES = ("organic", "stretch", "lenses")

    def __init__(self, cfg, tl, W: int, H: int, s: float, seed: int):
        self.cfg, self.tl, self.W, self.H, self.s = cfg, tl, W, H, s
        self.perlin = Perlin(seed)
        g = NOISE_GRID
        gx = np.arange(0, W + g, g, dtype=np.float64) / s   # coordinate in "px di riferimento"
        gy = np.arange(0, H + g, g, dtype=np.float64) / s
        self.GX, self.GY = np.meshgrid(gx, gy)
        self.bx, self.by = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.active = [n for n in self.STAGES if getattr(cfg, n).enabled]

    def __call__(self, mask: np.ndarray, k: int) -> np.ndarray:
        for name in self.active:
            mask = getattr(self, "_" + name)(mask, k)
        return mask

    def _up(self, a):
        return cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

    # ─── organic: ondulazioni morbide ────────────────────────────────────────
    def _organic(self, mask, k):
        tl = self.tl
        sc, t, amp = float(tl.organic_scale[k]), float(tl.organic_phase[k]), float(tl.organic_intensity[k] * self.s)
        X, Y = self.GX * sc, self.GY * sc
        nx = self.perlin.fbm(X, Y + t, octaves=4)
        ny = self.perlin.fbm(X + t, Y, octaves=4)
        mx = self.bx + self._up(nx) * amp
        my = self.by + self._up(ny) * amp
        return cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ─── stretch: stiramenti lunghi ──────────────────────────────────────────
    def _stretch(self, mask, k):
        c, tl = self.cfg.stretch, self.tl
        sc, t, inten = float(tl.stretch_scale[k]), float(tl.stretch_phase[k]), float(tl.stretch_intensity[k])
        X, Y = self.GX * sc, self.GY * sc
        wx = self.perlin.fbm(X + t, Y * 0.5, octaves=3) * (inten * 15 * c.horizontal)
        wy = self.perlin.fbm(X * 0.5, Y + t * 0.007, octaves=3) * (inten * 15 * c.vertical)
        if c.fine_detail > 0:
            fd = inten * c.fine_detail
            wx += self.perlin.fbm(X * 5 + t * 0.02, Y * 5, octaves=2, persistence=0.3) * fd
            wy += self.perlin.fbm(X * 5, Y * 5 + t * 0.02, octaves=2, persistence=0.3) * fd
        mx = np.clip(self.bx + self._up(wx) * self.s, 0, self.W - 1)
        my = np.clip(self.by + self._up(wy) * self.s, 0, self.H - 1)
        return cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ─── lenti: ogni lente lavora solo nel suo riquadro ──────────────────────
    def _lenses(self, mask, k):
        c, tl, s = self.cfg.lenses, self.tl, self.s
        pos, rad, ang = tl.lens_pos[k], tl.lens_radius[k], tl.lens_angle[k]
        strength = (tl.lens_strength[k] * tl.lens_mult[k]).astype(np.float32)
        wave_t = k * 0.03 + tl.lens_phase
        wave_amp, wave_freq, L = c.worm_wave * s, 0.01 / s, c.worm_length
        mx, my = self.bx.copy(), self.by.copy()

        for j in range(len(rad)):
            r = rad[j]
            R = math.hypot(r * L, r + wave_amp) if c.worm else r
            px, py = float(pos[j][0]), float(pos[j][1])
            x0, x1 = max(0, int(px - R)), min(self.W, int(px + R) + 2)
            y0, y1 = max(0, int(py - R)), min(self.H, int(py + R) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            dx = self.bx[y0:y1, x0:x1] - px
            dy = self.by[y0:y1, x0:x1] - py
            if c.worm:
                ca, sa = math.cos(float(ang[j])), math.sin(float(ang[j]))
                dxr = dx * ca - dy * sa
                dyr = dx * sa + dy * ca
                dys = dyr + np.sin(dxr * wave_freq + float(wave_t[j])) * wave_amp
                dist = np.sqrt((dxr / L) ** 2 + dys ** 2)
            else:
                dist = np.sqrt(dx * dx + dy * dy)
            disp = np.maximum(0.0, 1.0 - dist / float(r + 1e-6)) * float(strength[j])
            mx[y0:y1, x0:x1] += dx * disp
            my[y0:y1, x0:x1] += dy * disp

        return cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ─── supporto multiprocesso ──────────────────────────────────────────────────
_W = {}


def worker_init(deformer, mask0):
    cv2.setNumThreads(1)
    _W["d"], _W["m"] = deformer, mask0


def worker_deform(k):
    return _W["d"](_W["m"], k)
