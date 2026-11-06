"""
Timeline: tutto lo stato che evolve nel tempo, precalcolato prima di renderizzare.

Si simula a passi di 30 fps "di riferimento" (REF_FPS), così cambiare fps (es. --test)
non cambia la velocità del movimento. Il frame di output i usa il passo ref_index[i].
"""

import cv2
import numpy as np

from .effects.noise import Perlin

REF_FPS = 30


def ema(x: np.ndarray, smoothing: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        acc = acc * smoothing + v * (1.0 - smoothing)
        out[i] = acc
    return out


def _phase(speed: np.ndarray) -> np.ndarray:
    """Fase accumulata: con velocità costante è k*speed; con velocità variabile non salta."""
    return np.concatenate([[0.0], np.cumsum(speed)[:-1]])


class Timeline:
    def __init__(self, cfg, W, H, s, fps, duration, rng, bands=None, mask=None):
        self.n_ref = n = max(1, int(round(duration * REF_FPS)))
        self.n_out = max(1, int(round(duration * fps)))
        self.ref_index = np.minimum(np.round(np.arange(self.n_out) * REF_FPS / fps).astype(int), n - 1)

        # ─── audio: bassi e medi, 0-1, ammorbiditi ───────────────────────────
        # Ogni effetto ha la sua reattività (chiave `audio`, 0-1). L'audio cambia sempre
        # VELOCITÀ e AMPIEZZE, mai direttamente le posizioni: le fasi si accumulano,
        # così il movimento accelera e rallenta con la musica ma non salta mai.
        if bands is not None:
            sm = cfg.audio.smoothing
            self.bass, self.mid = ema(bands["bass"], sm), ema(bands["mid"], sm)
        else:
            self.bass = self.mid = np.zeros(n)

        # ─── organic / stretch ───────────────────────────────────────────────
        for name in ("organic", "stretch"):
            c = getattr(cfg, name)
            speed = 1.0 / (max(c.period, 1e-3) * REF_FPS)        # unità di rumore per frame
            setattr(self, f"{name}_gain", 1 + 0.6 * c.audio * self.bass)  # ampiezza: fino a +60%
            setattr(self, f"{name}_phase", _phase(speed * (1 + c.audio * self.bass)))  # velocità: fino a ×2

        self.glow = glow_curve(cfg.glow.intensity, n)

        self._simulate_lenses(cfg.lenses, W, H, s, n, rng, mask)

    # ═════════════════════════════════════════════════════════════════════════
    def _simulate_lenses(self, c, W, H, s, n, rng, mask=None):
        L = c.count if c.enabled else 0
        drift = c.motion == "drift"
        if c.motion not in ("paths", "drift"):
            raise ValueError(f"lenses.motion '{c.motion}' non valido: paths | drift")
        if not drift:
            horiz = ["horizontal_sweep", "horizontal_zigzag", "horizontal_wave_complex", "wave"]
            mixed = ["figure_eight", "spiral", "circular", "cross"]
            nh = int(L * 0.7)  # 70% percorsi orizzontali (seguono la scritta)
            kinds = [horiz[i % 4] for i in range(nh)] + [mixed[i % 4] for i in range(L - nh)]
            rng.shuffle(kinds)
            paths = np.stack([cinematic_path(W, H, kd, n) for kd in kinds]) if L else np.zeros((0, n, 2))
            off = rng.integers(0, n, L)
        base_r = (c.min_radius + (c.max_radius - c.min_radius) * rng.random(L) ** 2.5) * s
        base_s = rng.uniform(c.min_strength, c.max_strength, L)
        self.lens_angle = rng.uniform(0, 2 * np.pi, L).astype(np.float32)
        rng.uniform(-1, 1, L)  # (era la rotazione: estrazione tenuta per avere le stesse lenti a parità di seed)
        self.lens_phase = rng.uniform(0, 2 * np.pi, L)
        self.lens_radius = base_r.astype(np.float32)

        # i medi rendono le lenti più forti, i bassi più veloci
        self.lens_strength = (base_s[None, :] * (1 + 0.5 * c.audio * self.mid[:, None])).astype(np.float32)
        self.lens_pos = np.zeros((n, L, 2), np.float32)
        if L == 0:
            return

        step = c.speed / REF_FPS * s * (1 + c.audio * self.bass)  # px per frame di riferimento
        if drift:
            swarm = _Drift(c, W, H, s, rng, mask, L)
            pos = swarm.pos
        else:
            # punto sul percorso frame per frame (px percorsi = somma dei passi: niente salti)
            targets = _Tracks(paths, off).along(np.concatenate([[0.0], np.cumsum(step)[:-1]]))
            pos = targets[0].copy()
        vel = np.zeros((L, 2))
        inertia = min(0.99, c.inertia + 0.01)
        margin = c.min_radius * s
        pos[:, 0] = np.clip(pos[:, 0], margin, W - margin)  # dentro il frame fin dal primo frame
        pos[:, 1] = np.clip(pos[:, 1], margin, H - margin)

        for f in range(n):
            self.lens_pos[f] = pos
            if drift:
                # correnti lente + vagare personale + guinzaglio verso il logo
                desired = swarm.velocity(pos, f) * step[f]
                vel = vel * inertia + desired * (1 - inertia)
                pos = pos + vel
            else:
                # il punto sul percorso avanza a velocità costante; la lente lo insegue morbida
                pos = pos + (targets[min(f + 1, n - 1)] - pos) * (1 - inertia)
            pos[:, 0] = np.clip(pos[:, 0], margin, W - margin)
            pos[:, 1] = np.clip(pos[:, 1], margin, H - margin)


class _Tracks:
    """
    I percorsi delle lenti percorsi a velocità costante (in px), non "a indice".
    Ogni lente parte dal suo punto (off); i percorsi chiusi si ripetono, quelli
    aperti (da un lato all'altro) si percorrono avanti e indietro, senza salti.
    """

    def __init__(self, paths, off):
        self.paths = paths
        seg = np.linalg.norm(np.diff(paths, axis=1), axis=2)              # (L, n-1)
        self.cum = np.concatenate([np.zeros((len(paths), 1)), np.cumsum(seg, axis=1)], axis=1)
        self.total = np.maximum(self.cum[:, -1], 1e-6)
        gap = np.linalg.norm(paths[:, -1] - paths[:, 0], axis=1)
        self.closed = gap < np.maximum(3 * seg.mean(axis=1), 1.0)
        self.start = self.cum[np.arange(len(paths)), off]

    def along(self, d: np.ndarray) -> np.ndarray:
        """(frame, lente, xy): posizione di ogni lente dopo d[f] px percorsi."""
        out = np.empty((len(d), len(self.paths), 2))
        for j in range(len(self.paths)):
            T = self.total[j]
            s = self.start[j] + d
            if self.closed[j]:
                s = s % T
            else:
                s = s % (2 * T)
                s = np.where(s <= T, s, 2 * T - s)
            out[:, j, 0] = np.interp(s, self.cum[j], self.paths[j, :, 0])
            out[:, j, 1] = np.interp(s, self.cum[j], self.paths[j, :, 1])
        return out


class _Drift:
    """
    Movimento "organico" delle lenti (lenses.motion = "drift").

    • nascono sul logo (in punti a caso della sua sagoma, allargata di poco);
    • si lasciano trasportare da correnti lente: il campo di velocità è il curl di un
      rumore di Perlin, cioè un fluido senza sorgenti né pozzi → le lenti scorrono e
      girano come plancton, vicine si muovono simili, e non si ammassano;
    • ognuna aggiunge un po' del suo vagare (drift_turbulence);
    • se si allontanano dal logo un guinzaglio morbido le riporta indietro (drift_follow_logo).
    Tutte le misure sono in px di riferimento (risoluzione piena), così test e preview
    si muovono come il video finale.
    """

    Q = 4  # il campo "distanza dal logo" si calcola a 1/4 di risoluzione

    def __init__(self, c, W, H, s, rng, mask, L):
        self.c, self.s, self.W, self.H = c, s, W, H
        self.perlin = Perlin(int(rng.integers(1, 2**31)))
        self.own = rng.uniform(0, 1000, (L, 2))          # rumore personale di ogni lente
        q = self.Q
        w, h = max(1, W // q), max(1, H // q)
        if mask is None or not np.any(mask > 20):
            region = np.full((h, w), 255, np.uint8)
        else:
            small = cv2.resize(np.asarray(mask), (w, h), interpolation=cv2.INTER_AREA)
            r = max(1, int(round(c.max_radius * s * 0.35 / q)))
            region = cv2.dilate((small > 20).astype(np.uint8) * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ys, xs = np.nonzero(region)
        pick = rng.integers(0, len(xs), L)
        jitter = rng.uniform(0, q, (L, 2))
        self.pos = np.stack([xs[pick] * q + jitter[:, 0], ys[pick] * q + jitter[:, 1]], axis=1).astype(np.float64)
        # distanza dal logo (px di output) e direzione per tornarci
        dist = cv2.distanceTransform(255 - region, cv2.DIST_L2, 5) * q
        gy, gx = np.gradient(cv2.GaussianBlur(dist, (0, 0), 2))
        self.dist, self.gx, self.gy = dist, gx, gy

    def velocity(self, pos, f):
        """Velocità desiderata (circa unitaria) di ogni lente al frame di riferimento f."""
        c, s = self.c, self.s
        t = f * 0.004 * c.drift_change
        sc = 1.0 / max(1e-6, c.drift_scale * s)
        x, y = pos[:, 0] * sc, pos[:, 1] * sc
        e = 0.05
        psi = lambda a, b: self.perlin.fbm(a + t, b - 0.7 * t, octaves=2)
        vx = (psi(x, y + e) - psi(x, y - e)) / (2 * e)
        vy = -(psi(x + e, y) - psi(x - e, y)) / (2 * e)
        flow = np.stack([vx, vy], axis=1)
        # velocità ~costante, direzione dal flusso: |flow| va da 0 a ~3, la comprimo verso 1
        n = np.linalg.norm(flow, axis=1, keepdims=True)
        flow = flow / np.maximum(n, 1e-9) * np.tanh(n * 1.5)

        # vagare personale: una direzione che gira piano, diversa per ogni lente
        tt = f * 0.003 * c.drift_change
        a = self.perlin.noise(self.own[:, 0] + tt, self.own[:, 1]) * 2 * np.pi
        own = np.stack([np.cos(a), np.sin(a)], axis=1)
        tb = c.drift_turbulence
        v = flow * (1 - tb) + own * tb

        # guinzaglio verso il logo
        if c.drift_follow_logo > 0:
            q = self.Q
            ix = np.clip((pos[:, 0] / q).astype(int), 0, self.dist.shape[1] - 1)
            iy = np.clip((pos[:, 1] / q).astype(int), 0, self.dist.shape[0] - 1)
            d = self.dist[iy, ix]
            g = np.stack([self.gx[iy, ix], self.gy[iy, ix]], axis=1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            pull = np.clip(d / (40 * s), 0, 1)[:, None] * c.drift_follow_logo * 1.5
            v = v - g * pull
        return v


def glow_curve(intensity: float, n: int) -> np.ndarray:
    """Il glow pulsa una volta lungo tutto il video (+0.2 a metà)."""
    return intensity + np.sin(np.arange(n) / n * np.pi) * 0.2


def cinematic_path(W, H, kind, n):
    """Percorsi delle lenti (dalla v3), con forte preferenza orizzontale."""
    cx, cy = W // 2, H // 2
    hs, vs = 0.8, 0.2
    p = np.arange(n) / n
    if kind == "figure_eight":
        t = p * 4 * np.pi
        x, y = cx + W * hs * np.sin(t), cy + H * vs * np.sin(2 * t)
    elif kind == "spiral":
        t = p * 6 * np.pi
        k = 0.3 + 0.7 * np.sin(t * 0.3)
        x, y = cx + W * hs * k * np.cos(t), cy + H * vs * k * np.sin(t)
    elif kind == "wave":
        x, y = W * 0.05 + W * 0.9 * p, cy + np.sin(p * 8 * np.pi) * H * vs
    elif kind == "circular":
        t = p * 2 * np.pi
        x, y = cx + W * hs * np.cos(t), cy + H * vs * np.sin(t)
    elif kind == "cross":
        q = max(1, n // 4)
        i = np.arange(n)
        x = np.select([i < q, i < 2 * q, i < 3 * q], [W * 0.05 + W * hs * (i / q), cx + W * hs * ((i - q) / q), np.full(n, cx)], np.full(n, cx))
        y = np.select([i < 2 * q, i < 3 * q], [np.full(n, cy), cy - H * vs * ((i - 2 * q) / q)], cy - H * vs + 2 * H * vs * ((i - 3 * q) / q))
    elif kind == "horizontal_sweep":
        x = W * 0.05 + W * 0.9 * (0.5 + 0.5 * np.sin(p * 2 * np.pi)) + W * 0.1 * np.sin(p * 8 * np.pi)
        y = cy + H * 0.08 * np.sin(p * 12 * np.pi) * np.cos(p * 4 * np.pi)
    elif kind == "horizontal_zigzag":
        x, y = W * 0.1 + W * 0.8 * p, cy + H * 0.15 * np.sin(p * 16 * np.pi)
    elif kind == "horizontal_wave_complex":
        x = W * 0.05 + W * 0.9 * p + W * 0.05 * np.sin(p * 20 * np.pi)
        y = cy + np.sin(p * 6 * np.pi) * H * 0.1 + np.sin(p * 15 * np.pi) * H * 0.05 + np.cos(p * 25 * np.pi) * H * 0.03
    else:
        raise ValueError(kind)
    return np.stack([x, y], axis=1).astype(np.float64)
