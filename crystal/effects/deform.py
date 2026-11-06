"""
Deformazioni della maschera del logo: organic → stretch → lenti.

Tutto quello che cambia nel tempo è precalcolato nella Timeline, quindi deformare
il frame k è una funzione pura (k) → maschera: si può fare in parallelo.

Come funziona (ed è il motivo per cui è veloce):
  ogni deformazione è un campo di spostamenti. Invece di applicare tre `remap`
  uno dopo l'altro alla maschera (tre interpolazioni a piena risoluzione) si
  compongono le MAPPE e si fa un solo `remap` finale:

      p = x + lenti(x)            (l'ultima deformazione applicata si legge per prima)
      q = p + stretch(p)
      r = q + organic(q)
      maschera_finale(x) = maschera(r)

  È lo stesso risultato dei tre remap in sequenza, ma il logo viene interpolato
  una volta sola (bordi più puliti).

  • La mappa si calcola su una griglia a passo WORK_GRID px (a piena risoluzione)
    e solo alla fine si porta a risoluzione piena: è un campo continuo, a 2 px la
    differenza non si vede e costa 4 volte meno.
  • organic e stretch sono rumore di Perlin calcolato su una griglia ancora più
    rada, con passo scelto in base alla frequenza del rumore (onde larghe → pochi
    punti), e letto con interpolazione.

Per aggiungere una deformazione: scrivi un metodo che restituisce lo spostamento
e componilo in `maps()` (+ la sua sezione in defaults.toml).
"""

import math

import cv2
import numpy as np

from .noise import Perlin

WORK_GRID = 2      # px (a piena risoluzione) tra i punti in cui si calcola la mappa
NOISE_PTS = 6      # punti di griglia per periodo dell'ottava più fine del rumore


class _NoiseGrid:
    """Griglia rada per un campo di rumore: passo g px, coordinate in px di riferimento."""

    def __init__(self, W, H, s, max_freq):
        period_px = s / max(max_freq, 1e-9)  # periodo dell'ottava più fine, in px di output
        self.g = g = int(np.clip(period_px / NOISE_PTS, 2, 24))
        gx = np.arange(W // g + 3, dtype=np.float64) * g / s
        gy = np.arange(H // g + 3, dtype=np.float64) * g / s
        self.X, self.Y = np.meshgrid(gx, gy)

    def sample(self, field, mx, my):
        return cv2.remap(field, mx * (1 / self.g), my * (1 / self.g), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)


class Deformer:
    STAGES = ("organic", "stretch", "lenses")

    def __init__(self, cfg, tl, W: int, H: int, s: float, seed: int):
        self.cfg, self.tl, self.W, self.H, self.s = cfg, tl, W, H, s
        self.perlin = Perlin(seed)
        self.active = [n for n in self.STAGES if getattr(cfg, n).enabled]

        # griglia di lavoro: punti al centro di blocchi ls×ls → il resize finale è esatto
        ls = self.ls = max(1, int(round(WORK_GRID * s)))
        self.lw, self.lh = -(-W // ls), -(-H // ls)
        self.gx = np.arange(self.lw, dtype=np.float32) * ls + (ls - 1) / 2
        self.gy = np.arange(self.lh, dtype=np.float32) * ls + (ls - 1) / 2
        self.bx, self.by = np.meshgrid(self.gx, self.gy)

        # frequenza dell'ottava più fine di ogni campo (×1.3 di margine)
        o, st = cfg.organic, cfg.stretch
        self.ngrid = {
            "organic": _NoiseGrid(W, H, s, 1.3 * 2 ** 3 / o.size),
            "stretch": _NoiseGrid(W, H, s, 1.3 * 2 ** 2 / st.size),
        }

    # la griglia di lavoro non si spedisce ai processi: si ricrea lì (pickle più leggero)
    def __getstate__(self):
        d = self.__dict__.copy()
        del d["bx"], d["by"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.bx, self.by = np.meshgrid(self.gx, self.gy)

    def __call__(self, mask: np.ndarray, k: int) -> np.ndarray:
        if not self.active:
            return mask
        mx, my = self.maps(k)
        return cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def maps(self, k: int):
        """Mappa (mx, my) a piena risoluzione: il pixel x prende il valore della maschera in (mx, my)."""
        mx, my = self.bx, self.by
        if "lenses" in self.active:
            lx, ly = self._lenses(k)
            mx, my = mx + lx, my + ly
        if "stretch" in self.active:
            dx, dy = cv2.split(self.ngrid["stretch"].sample(self._stretch(k), mx, my))
            mx = np.clip(mx + dx, 0, self.W - 1)
            my = np.clip(my + dy, 0, self.H - 1)
        if "organic" in self.active:
            dx, dy = cv2.split(self.ngrid["organic"].sample(self._organic(k), mx, my))
            mx, my = mx + dx, my + dy
        ls = self.ls
        if ls > 1:
            size = (self.lw * ls, self.lh * ls)
            mx = cv2.resize(mx, size, interpolation=cv2.INTER_LINEAR)[:self.H, :self.W]
            my = cv2.resize(my, size, interpolation=cv2.INTER_LINEAR)[:self.H, :self.W]
        return mx, my

    # ─── organic: ondulazioni morbide → campo (dx, dy) sulla griglia rada ────
    def _organic(self, k):
        c, tl, ng = self.cfg.organic, self.tl, self.ngrid["organic"]
        sc, t, amp = 1.0 / c.size, float(tl.organic_phase[k]), float(c.amount * tl.organic_gain[k] * self.s)
        X, Y = ng.X * sc, ng.Y * sc
        f = np.empty((*X.shape, 2), np.float32)
        f[..., 0] = self.perlin.fbm(X, Y + t, octaves=4) * amp
        f[..., 1] = self.perlin.fbm(X + t, Y, octaves=4) * amp
        return f

    # ─── stretch: stiramenti lunghi ──────────────────────────────────────────
    def _stretch(self, k):
        c, tl, ng = self.cfg.stretch, self.tl, self.ngrid["stretch"]
        sc, t, gain = 1.0 / c.size, float(tl.stretch_phase[k]), float(tl.stretch_gain[k] * self.s)
        X, Y = ng.X * sc, ng.Y * sc
        f = np.empty((*X.shape, 2), np.float32)
        # campi allungati (una coordinata a metà frequenza) che scorrono nella loro direzione
        f[..., 0] = self.perlin.fbm(X + t, Y * 0.5, octaves=3) * (c.horizontal * gain)
        f[..., 1] = self.perlin.fbm(X * 0.5, Y + t, octaves=3) * (c.vertical * gain)
        return f

    # ─── lenti: ogni lente lavora solo nel suo riquadro ──────────────────────
    def _lenses(self, k):
        c, tl, s, ls = self.cfg.lenses, self.tl, self.s, self.ls
        pos, rad, ang = tl.lens_pos[k], tl.lens_radius, tl.lens_angle
        strength = tl.lens_strength[k]
        wave_t = k * 0.03 + tl.lens_phase
        wave_amp, wave_freq, L = c.worm_wave * s, 0.01 / s, c.worm_length
        lx = np.zeros((self.lh, self.lw), np.float32)
        ly = np.zeros((self.lh, self.lw), np.float32)
        gxs, gys = self.gx, self.gy

        for j in range(len(rad)):
            r = float(rad[j])
            R = math.hypot(r * L, r + wave_amp) if c.worm else r
            px, py = float(pos[j][0]), float(pos[j][1])
            x0, x1 = max(0, int((px - R) / ls)), min(self.lw, int((px + R) / ls) + 2)
            y0, y1 = max(0, int((py - R) / ls)), min(self.lh, int((py + R) / ls) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            dx = (gxs[x0:x1] - px)[None, :]
            dy = (gys[y0:y1] - py)[:, None]
            if c.worm:
                ca, sa = math.cos(float(ang[j])), math.sin(float(ang[j]))
                dxr = dx * ca - dy * sa
                dys = dx * sa + dy * ca + np.sin(dxr * wave_freq + float(wave_t[j])) * wave_amp
                dist = np.sqrt((dxr * (1 / L)) ** 2 + dys * dys)
            else:
                dist = np.sqrt(dx * dx + dy * dy)
            disp = np.maximum(0.0, 1.0 - dist * (1 / (r + 1e-6))) * float(strength[j])
            lx[y0:y1, x0:x1] += dx * disp
            ly[y0:y1, x0:x1] += dy * disp
        return lx, ly


# ─── supporto multiprocesso ──────────────────────────────────────────────────
_W = {}


def worker_init(deformer, mask0):
    from ..procs import watch_parent
    watch_parent()  # se il render muore, il worker esce invece di restare appeso
    cv2.setNumThreads(1)
    _W["d"], _W["m"] = deformer, mask0


def worker_deform(k):
    return _W["d"](_W["m"], k)
