"""
Perlin noise 2D vettorizzato (numpy). Sostituisce `noise.pnoise2` chiamato pixel per pixel
nei loop Python della v3 — era il motivo principale dei render lentissimi.

`fbm` è normalizzato come pnoise2(octaves=...) della libreria `noise`.
"""

import numpy as np

_GRAD = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


class Perlin:
    def __init__(self, seed: int):
        p = np.random.default_rng(seed).permutation(256)
        self.perm = np.concatenate([p, p]).astype(np.int32)

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.floor(x)
        y0 = np.floor(y)
        xf = (x - x0).astype(np.float32)
        yf = (y - y0).astype(np.float32)
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        p = self.perm
        pa, pb = p[xi], p[xi + 1]
        g = lambda h, dx, dy: _GRAD[h & 7, 0] * dx + _GRAD[h & 7, 1] * dy
        n00 = g(p[pa + yi], xf, yf)
        n01 = g(p[pa + yi + 1], xf, yf - 1)
        n10 = g(p[pb + yi], xf - 1, yf)
        n11 = g(p[pb + yi + 1], xf - 1, yf - 1)
        u, v = _fade(xf), _fade(yf)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return nx0 + v * (nx1 - nx0)

    def fbm(self, x, y, octaves=1, persistence=0.5, lacunarity=2.0):
        total = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), np.float32)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += self.noise(x * freq, y * freq) * amp
            norm += amp
            freq *= lacunarity
            amp *= persistence
        return total / norm
