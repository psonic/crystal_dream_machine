"""
Composizione del frame: scie (tracers), texture, glow, fusione logo/sfondo.
Formule delle blend mode e della fusione avanzata riprese dalla v3.
Tutti i calcoli sono in float32 0-1.
"""

import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  Blend mode
# ═════════════════════════════════════════════════════════════════════════════

def blend_mode(base, blend, mode):
    base = np.clip(base, 0, 1)
    blend = np.clip(blend, 0, 1)
    if mode == "normal":
        return blend
    if mode == "multiply":
        return base * blend
    if mode == "screen":
        return 1 - (1 - base) * (1 - blend)
    if mode == "overlay":
        return np.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
    if mode == "soft_light":
        return np.where(blend < 0.5, base - (1 - 2 * blend) * base * (1 - base),
                        base + (2 * blend - 1) * (np.sqrt(base) - base))
    if mode == "hard_light":
        return np.where(blend < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
    if mode == "color_dodge":
        return np.minimum(1, base / (1 - blend + 1e-6))
    if mode == "color_burn":
        return 1 - np.minimum(1, (1 - base) / (blend + 1e-6))
    if mode == "darken":
        return np.minimum(base, blend)
    if mode == "lighten":
        return np.maximum(base, blend)
    if mode == "difference":
        return np.abs(base - blend)
    if mode == "exclusion":
        return base + blend - 2 * base * blend
    raise ValueError(f"blend mode sconosciuta: {mode}")


BLEND_MODES = ["normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
               "color_dodge", "color_burn", "darken", "lighten", "difference", "exclusion"]


def odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


# ═════════════════════════════════════════════════════════════════════════════
#  Scie
# ═════════════════════════════════════════════════════════════════════════════

class Trail:
    """
    Scia a decadimento esponenziale: un solo buffer invece di sommare N frame passati
    a ogni frame (la v3 ne sommava 85+105 → lentissimo). Stessa luminosità a regime;
    la tinta ruota nel tempo, quindi lungo la scia si vede la stessa sfumatura di prima.
    """

    def __init__(self, c, fps):
        n = max(1.0, c.trail * fps / 30)
        self.decay = max(0.0, 1 - 2 / n) if n > 2 else 0.0
        self.opacity = c.opacity
        self.hue_step = c.hue_shift  # per frame di riferimento (30 fps)
        r, g, b = c.color
        self.hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        self.buf = None

    def color(self, i):
        h = (self.hsv[0] - i * self.hue_step) % 180
        hsv = np.uint8([[[h, self.hsv[1], self.hsv[2]]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255.0

    def push(self, edges, i, mult=1.0):
        if self.buf is None:
            self.buf = np.zeros((*edges.shape, 3), np.float32)
        else:
            self.buf *= self.decay
        self.buf[edges > 0] += self.color(i) * (self.opacity * mult)


# ═════════════════════════════════════════════════════════════════════════════
#  Texture
# ═════════════════════════════════════════════════════════════════════════════

def apply_texture(base, tex, alpha, mode, mask=None):
    out = base * (1 - alpha) + blend_mode(base, tex, mode) * alpha
    if mask is not None:
        m = mask[..., None]
        out = base * (1 - m) + out * m
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Fusione logo / sfondo (la "advanced blending" della v3)
# ═════════════════════════════════════════════════════════════════════════════

def mask_roi(mask_u8, pad):
    """Riquadro (y0,y1,x0,x1) che contiene la maschera + pad, o None se vuota."""
    x, y, w, h = cv2.boundingRect(mask_u8)
    if w == 0:
        return None
    H, W = mask_u8.shape
    return max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)


def blend_logo(frame, mask_u8, color, c, s, logo_img=None):
    """
    Fonde il logo con lo sfondo (in place su `frame`, float32 0-1 BGR).
    color: colore BGR 0-1 del logo; logo_img: immagine del logo (se c'è la texture).
    Calcola solo i pixel vicini al logo: il logo è fatto di tratti, il resto del frame non cambia.
    """
    pad = odd(c.edge_softness * s) + odd(c.edge_blur * s) + 4
    roi = mask_roi(mask_u8, pad)
    if roi is None:
        return frame
    y0, y1, x0, x1 = roi
    m_u8 = mask_u8[y0:y1, x0:x1]
    soft = cv2.GaussianBlur(m_u8, (odd(c.edge_softness * s),) * 2, 0)
    region = soft > 0
    e2d = None
    if c.edge_detection:
        k = max(1, int(c.edge_blur * s) // 3)
        edges = cv2.dilate(cv2.Canny(m_u8, 50, 150), np.ones((k, k), np.uint8), iterations=2)
        e2d = cv2.GaussianBlur(edges, (odd(c.edge_blur * s),) * 2, 0)
        region |= e2d > 0

    idx = np.flatnonzero(region)
    roi_img = np.ascontiguousarray(frame[y0:y1, x0:x1])
    flat = roi_img.reshape(-1, 3)
    B = flat[idx]                                       # sfondo, (N,3)
    S = (soft.ravel()[idx].astype(np.float32) * (1 / 255))[:, None]
    inside = m_u8.ravel()[idx] > 0
    if logo_img is not None:
        Lg = np.ascontiguousarray(logo_img[y0:y1, x0:x1]).reshape(-1, 3)[idx]
    else:
        Lg = inside[:, None] * color.astype(np.float32)

    BL = Lg.copy()
    if c.adaptive and inside.any():
        avg_bg = B[inside].mean(axis=0)
        if c.harmonize:
            BL[inside] = BL[inside] * (1 - c.color_strength) + avg_bg * c.color_strength
        if c.luminance_match:
            lw = np.array([0.114, 0.587, 0.299], np.float32)  # BGR
            lum_logo = float((BL[inside] @ lw).mean())
            if lum_logo > 0:
                BL[inside] *= 0.5 + 0.5 * float((B[inside] @ lw).mean()) / lum_logo

    res = B * S
    res = res * (1 - c.strength) + blend_mode(res, BL * S, c.mode) * c.strength
    if c.transparency > 0:
        res = res * (1 - c.transparency) + B * S * c.transparency
    final = B * (1 - S) + res

    if e2d is not None:
        E = (e2d.ravel()[idx].astype(np.float32) * (1 / 255))[:, None]
        st = c.strength * 1.5
        edge_bl = B * (1 - st) + blend_mode(B, Lg, c.mode) * st
        final = final * (1 - E) + edge_bl * E * S

    flat[idx] = final
    frame[y0:y1, x0:x1] = roi_img
    return frame
