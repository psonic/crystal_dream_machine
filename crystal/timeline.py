"""
Timeline: tutto lo stato che evolve nel tempo, precalcolato prima di renderizzare.

Si simula a passi di 30 fps "di riferimento" (REF_FPS), così cambiare fps (es. --test)
non cambia la velocità del movimento. Il frame di output i usa il passo ref_index[i].
"""

import numpy as np

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
    def __init__(self, cfg, W, H, s, fps, duration, rng, bands=None):
        self.n_ref = n = max(1, int(round(duration * REF_FPS)))
        self.n_out = max(1, int(round(duration * fps)))
        self.ref_index = np.minimum(np.round(np.arange(self.n_out) * REF_FPS / fps).astype(int), n - 1)
        k = np.arange(n, dtype=np.float64)

        # ─── audio → fattori per le lenti ────────────────────────────────────
        a = cfg.audio
        if bands is not None:
            sm = {b: ema(bands[b], a.smoothing) for b in ("bass", "mid", "high")}
            self.speed_f = np.clip(1 + sm["bass"] * a.bass, 0.5, 1.5)
            self.strength_f = np.clip(1 + sm["mid"] * a.mid, 0.5, 1.5)
            self.puls_f = np.clip(1 + sm["high"] * a.high, 0.5, 1.5)
        else:
            self.speed_f = self.strength_f = self.puls_f = np.ones(n)

        # ─── organic / stretch (eventualmente audio-reattivi) ────────────────
        for name in ("organic", "stretch"):
            c = getattr(cfg, name)
            inten = np.full(n, c.intensity, np.float64)
            speed = np.full(n, c.speed, np.float64)
            scale = np.full(n, c.scale, np.float64)
            if c.audio_reactive and bands is not None:
                inten = np.clip(ema(c.intensity + bands["bass"] * c.bass_intensity, c.smoothing),
                                c.intensity * 0.6, c.intensity * 1.4)
                speed = np.clip(ema(c.speed + bands["bass"] * c.bass_speed, c.smoothing),
                                c.speed * 0.7, c.speed * 1.5)
                scale = np.clip(ema(c.scale + bands["mid"] * c.mid_scale, c.smoothing),
                                c.scale * 0.8, c.scale * 1.3)
            setattr(self, f"{name}_intensity", inten)
            setattr(self, f"{name}_scale", scale)
            setattr(self, f"{name}_phase", _phase(speed))

        # ─── variazioni lente automatiche ────────────────────────────────────
        v = cfg.variation
        if v.enabled:
            seed_t = k * 0.001
            self.lens_mult = 1 + np.cos(seed_t * v.speed_medium + 5.5) * v.amplitude
            self.tracer_mult = 1 + np.sin(seed_t * v.speed_fast + 2.0) * v.amplitude
            self.bg_tracer_mult = 1 + np.cos(seed_t * v.speed_fast + 6.0) * v.amplitude
        else:
            self.lens_mult = self.tracer_mult = self.bg_tracer_mult = np.ones(n)

        self.glow = glow_curve(cfg.glow.intensity, n)

        self._simulate_lenses(cfg.lenses, W, H, s, n, rng)

    # ═════════════════════════════════════════════════════════════════════════
    def _simulate_lenses(self, c, W, H, s, n, rng):
        L = c.count if c.enabled else 0
        horiz = ["horizontal_sweep", "horizontal_zigzag", "horizontal_wave_complex", "wave"]
        mixed = ["figure_eight", "spiral", "circular", "cross"]
        nh = int(L * 0.7)  # 70% percorsi orizzontali (seguono la scritta)
        kinds = [horiz[i % 4] for i in range(nh)] + [mixed[i % 4] for i in range(L - nh)]
        rng.shuffle(kinds)

        paths = np.stack([cinematic_path(W, H, kd, n) for kd in kinds]) if L else np.zeros((0, n, 2))
        off = rng.integers(0, n, L)
        base_r = (c.min_radius + (c.max_radius - c.min_radius) * rng.random(L) ** 2.5) * s
        base_s = rng.uniform(c.min_strength, c.max_strength, L)
        angle = rng.uniform(0, 2 * np.pi, L)
        rot = rng.uniform(c.rotation_min, c.rotation_max, L)
        self.lens_phase = rng.uniform(0, 2 * np.pi, L)

        pos = paths[np.arange(L), off].astype(np.float64)
        vel = np.zeros((L, 2))
        inertia = min(0.99, c.inertia + 0.01)
        margin = c.min_radius * s

        self.lens_pos = np.zeros((n, L, 2), np.float32)
        self.lens_radius = np.zeros((n, L), np.float32)
        self.lens_strength = np.zeros((n, L), np.float32)
        self.lens_angle = np.zeros((n, L), np.float32)
        if L == 0:
            return

        for f in range(n):
            pt = (f * c.pulsation_speed + self.lens_phase) * self.puls_f[f]
            radius = base_r * (1 + c.pulsation_amplitude * self.puls_f[f] * np.sin(pt) * 0.5) if c.pulsation else base_r
            strength = (base_s * (1 + c.force_pulsation_amplitude * self.strength_f[f] * np.sin(pt * 1.2) * 0.3)
                        if c.force_pulsation else base_s)
            self.lens_pos[f], self.lens_radius[f] = pos, radius
            self.lens_strength[f], self.lens_angle[f] = strength, angle

            # movimento lungo il percorso con inerzia
            prog = ((f + off) * c.path_speed * self.speed_f[f]) % n
            i0 = prog.astype(int)
            i1 = (i0 + 1) % n
            fr = prog - i0
            smooth = (3 * fr ** 2 - 2 * fr ** 3)[:, None]
            a, b = paths[np.arange(L), i0], paths[np.arange(L), i1]
            target = a + (b - a) * smooth
            d = target - pos
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            desired = np.where(dist > 0, d / np.maximum(dist, 1e-9), 0) * (c.speed * s * self.speed_f[f])
            vel = vel * inertia + desired * (1 - inertia)
            pos = pos + vel
            angle = angle + rot * c.rotation_speed
            pos[:, 0] = np.clip(pos[:, 0], margin, W - margin)
            pos[:, 1] = np.clip(pos[:, 1], margin, H - margin)


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
