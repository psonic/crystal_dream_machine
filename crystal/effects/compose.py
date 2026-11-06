"""
Composizione del frame: scie (tracers), texture, glow, fusione logo/sfondo.
Formule delle blend mode e della fusione avanzata riprese dalla v3.
Tutti i calcoli sono in float32 0-1, BGR.

Per la velocità: le operazioni su tutto il frame usano OpenCV (multithread),
quelle che riguardano solo il logo lavorano sui soli pixel del logo.
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


def scalar(c):
    """Colore/valore → Scalar OpenCV a 4 componenti (per operazioni con costanti su immagini a 3 canali)."""
    c = np.broadcast_to(np.asarray(c, np.float64), (3,))
    return (float(c[0]), float(c[1]), float(c[2]), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Scie
# ═════════════════════════════════════════════════════════════════════════════

class Trail:
    """
    Scia a decadimento esponenziale: un solo buffer invece di sommare N frame passati
    a ogni frame (la v3 ne sommava 85+105 → lentissimo). Stessa luminosità a regime;
    la tinta ruota nel tempo, quindi lungo la scia si vede la stessa sfumatura di prima.

    `push` non modifica il buffer ricevuto ma ne restituisce uno nuovo: così lo stato
    delle scie "prima del frame i" resta disponibile (la preview lo usa per ricomporre
    il frame corrente quando cambi un parametro).
    """

    def __init__(self, c, fps):
        n = max(1.0, c.trail * fps / 30)
        self.decay = max(0.0, 1 - 2 / n) if n > 2 else 0.0
        self.opacity = c.opacity
        self.hue_step = c.hue_shift  # per frame di riferimento (30 fps)
        r, g, b = c.color
        self.hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    def color(self, i):
        h = (self.hsv[0] - i * self.hue_step) % 180
        hsv = np.uint8([[[h, self.hsv[1], self.hsv[2]]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255.0

    def push(self, buf, edges, i, binary=True):
        """edges: bordi binari (0/255) oppure, con binary=False, copertura 0-255 per pixel."""
        if buf is None:
            out = np.zeros((*edges.shape, 3), np.float32)
        else:
            out = cv2.multiply(buf, scalar(self.decay))
        col = self.color(i) * self.opacity
        if binary:
            cv2.add(out, scalar(col), dst=out, mask=edges)
        else:
            cv2.add(out, cv2.multiply(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), scalar(col / 255), dtype=cv2.CV_32F),
                    dst=out)
        return out


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
#  Glow
# ═════════════════════════════════════════════════════════════════════════════

def add_glow(frame, mask_u8, color, size, intensity):
    """Bagliore del colore del logo attorno alla maschera (in place, solo nel riquadro del logo)."""
    ks = odd(size)
    roi = mask_roi(mask_u8, ks)
    if not roi or intensity <= 0:
        return
    y0, y1, x0, x1 = roi
    glow = cv2.GaussianBlur(mask_u8[y0:y1, x0:x1], (ks, ks), 0)
    g3 = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
    sub = frame[y0:y1, x0:x1]
    sub += cv2.multiply(g3, scalar(color * (intensity / 255)), dtype=cv2.CV_32F)
    cv2.threshold(sub, 1.0, 1.0, cv2.THRESH_TRUNC, dst=sub)


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


def paste_logo(frame, mask_u8, color, logo_img=None):
    """Logo incollato sopra, senza fusione (in place)."""
    roi = mask_roi(mask_u8, 1)
    if not roi:
        return
    y0, y1, x0, x1 = roi
    m = cv2.cvtColor(mask_u8[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR).astype(np.float32) * (1 / 255)
    logo = logo_img[y0:y1, x0:x1] if logo_img is not None else np.asarray(color, np.float32)
    sub = frame[y0:y1, x0:x1]
    sub *= 1 - m
    sub += logo * m


def blend_logo(frame, mask_u8, color, c, s, logo_img=None):
    """
    Fonde il logo con lo sfondo (in place su `frame`, float32 0-1 BGR, contiguo).
    color: colore BGR 0-1 del logo; logo_img: immagine del logo (se c'è la texture).
    Calcola solo i pixel vicini al logo: il logo è fatto di tratti, il resto del frame non cambia.
    """
    pad = odd(c.edge_softness * s) + odd(c.edge_blur * s) + 4
    roi = mask_roi(mask_u8, pad)
    if roi is None:
        return frame
    y0, y1, x0, x1 = roi
    m_u8 = mask_u8[y0:y1, x0:x1]
    ks = odd(c.edge_softness * s)
    soft = cv2.GaussianBlur(m_u8, (ks, ks), 0) if ks > 1 else m_u8
    e2d = None
    if c.edge_detection:
        k = max(1, int(c.edge_blur * s) // 3)
        edges = cv2.dilate(cv2.Canny(m_u8, 50, 150), np.ones((k, k), np.uint8), iterations=2)
        e2d = cv2.GaussianBlur(edges, (odd(c.edge_blur * s),) * 2, 0)
        region = cv2.bitwise_or(soft, e2d)
    else:
        region = soft

    # indici (nel frame intero) dei pixel da toccare: niente copie della regione
    pts = cv2.findNonZero(region)
    if pts is None:
        return frame
    pts = pts.reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    W = frame.shape[1]
    idx = (ys + y0) * W + (xs + x0)
    flat = frame.reshape(-1, 3)
    B = flat[idx]                                       # sfondo, (N,3)
    S = soft[ys, xs].astype(np.float32)[:, None] * (1 / 255)
    inside = m_u8[ys, xs] > 0
    all_inside = region is m_u8
    lw = np.array([0.114, 0.587, 0.299], np.float32)    # luminanza, BGR

    if logo_img is not None:
        Lg = logo_img.reshape(-1, 3)[idx]
        BL = Lg.copy()
        if inside.any():
            Bi = B[inside]
            avg_bg = Bi.mean(axis=0)
            if c.harmonize:
                BL[inside] = BL[inside] * (1 - c.color_strength) + avg_bg * c.color_strength
            if c.luminance_match:
                lum_logo = float((BL[inside] @ lw).mean())
                if lum_logo > 0:
                    BL[inside] *= 0.5 + 0.5 * float((Bi @ lw).mean()) / lum_logo
    else:
        # logo a tinta unita: armonizzazione e luminanza danno un colore costante → 3 numeri
        col = color.astype(np.float32)
        if inside.any():
            avg_bg = (B if all_inside else B[inside]).mean(axis=0)
            if c.harmonize:
                col = col * (1 - c.color_strength) + avg_bg * c.color_strength
            if c.luminance_match:
                lum_logo = float(col @ lw)
                if lum_logo > 0:
                    col = col * (0.5 + 0.5 * float(avg_bg @ lw) / lum_logo)
        ins = inside[:, None].astype(np.float32)
        Lg = ins * color.astype(np.float32) if e2d is not None else None  # serve solo per i bordi
        BL = ins * col.astype(np.float32)

    BS = B * S
    res = BS * (1 - c.strength) + blend_mode(BS, BL * S, c.mode) * c.strength
    final = B - BS + res

    if e2d is not None:
        E = e2d[ys, xs].astype(np.float32)[:, None] * (1 / 255)
        st = c.strength * 1.5
        edge_bl = B * (1 - st) + blend_mode(B, Lg, c.mode) * st
        final = final * (1 - E) + edge_bl * E * S

    flat[idx] = final
    return frame
