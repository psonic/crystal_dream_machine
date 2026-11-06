"""
Configurazione: defaults.toml  →  preset (presets/<nome>.toml o un .json salvato)  →  --set chiave=valore

Il risultato è un albero di namespace: cfg.lenses.count, cfg.tracers.logo.opacity, ...
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
PRESETS_DIR = ROOT_DIR / "presets"
DEFAULTS_FILE = PKG_DIR / "defaults.toml"

FORMATS = {
    "ig_story": (1080, 1920),
    "ig_post": (1080, 1350),
    "square": (1080, 1080),
    "landscape": (1920, 1080),
}

# Preset di fusione logo/sfondo (dalla v3). Valgono come base: le chiavi scritte
# nel preset dell'utente li sovrascrivono.
BLEND_PRESETS = {
    "cinematic":   dict(mode="overlay",     strength=0.56,   edge_detection=False, edge_blur=3, harmonize=True,  luminance_match=False, color_strength=0.9),
    "artistic":    dict(mode="difference",  strength=0.81,   edge_detection=True,  edge_blur=5, harmonize=False, luminance_match=False, color_strength=0.2),
    "soft":        dict(mode="soft_light",  strength=0.36,   edge_detection=False, edge_blur=3, harmonize=True,  luminance_match=True,  color_strength=0.5),
    "dramatic":    dict(mode="multiply",    strength=0.7225, edge_detection=True,  edge_blur=3, harmonize=False, luminance_match=True,  color_strength=0.3),
    "bright":      dict(mode="screen",      strength=0.49,   edge_detection=False, edge_blur=5, harmonize=True,  luminance_match=False, color_strength=0.4),
    "intense":     dict(mode="hard_light",  strength=0.81,   edge_detection=False, edge_blur=3, harmonize=False, luminance_match=False, color_strength=0.25),
    "psychedelic": dict(mode="exclusion",   strength=0.9025, edge_detection=True,  edge_blur=3, harmonize=False, luminance_match=False, color_strength=0.1),
    "glow":        dict(mode="color_dodge", strength=0.6,    edge_detection=False, edge_blur=3, harmonize=True,  luminance_match=False, color_strength=0.35),
    "dark":        dict(mode="color_burn",  strength=0.6,    edge_detection=True,  edge_blur=3, harmonize=False, luminance_match=True,  color_strength=0.4),
    "geometric":   dict(mode="normal",      strength=1.0,    edge_detection=True,  edge_blur=1, harmonize=False, luminance_match=False, color_strength=0.0),
}

# ─── Chiavi vecchie (v4.0, ottobre 2026) → attuali ───────────────────────────
# I preset e i .json salvati accanto ai video con i nomi vecchi si caricano ancora:
# vengono convertiti al volo, con lo stesso risultato dove il comportamento non è cambiato.
_OLD = {
    "organic": dict(speed=0.008, scale=0.001, intensity=20.0, audio_reactive=True, bass_speed=0.003),
    "stretch": dict(speed=0.0003, scale=0.0015, intensity=25.0, horizontal=0.6, vertical=0.01,
                    audio_reactive=False, bass_speed=0.0025),
}
_OLD_WAVE = {"speed", "scale", "intensity", "horizontal", "vertical", "fine_detail", "audio_reactive",
             "bass_intensity", "bass_speed", "mid_scale", "smoothing"}
_OLD_LENS = {"path_speed", "drift_speed", "rotation_min", "rotation_max", "rotation_speed", "pulsation",
             "pulsation_speed", "pulsation_amplitude", "force_pulsation", "force_pulsation_amplitude"}


def _sub(d: dict, *path) -> dict:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d if isinstance(d, dict) else {}


def _is_old(u: dict) -> bool:
    """Un file col formato vecchio ha almeno una chiave che oggi non esiste più."""
    return bool(
        "variation" in u
        or {"darken", "contrast"} & _sub(u, "background").keys()
        # (horizontal/vertical esistono anche oggi, in px: da sole non dicono niente)
        or (_OLD_WAVE - {"horizontal", "vertical"}) & (_sub(u, "organic").keys() | _sub(u, "stretch").keys())
        or _OLD_LENS & _sub(u, "lenses").keys()
        or {"bass", "mid", "high"} & _sub(u, "audio").keys()
        or {"threshold1", "threshold2"} & (_sub(u, "tracers", "logo").keys() | _sub(u, "tracers", "background").keys())
        or {"transparency", "adaptive"} & _sub(u, "blend").keys()
    )


def migrate(user: dict) -> tuple[dict, list[str]]:
    """Converte le chiavi vecchie. Ritorna (config, elenco delle conversioni fatte)."""
    if not _is_old(user):
        return user, []
    u = copy.deepcopy(user)
    notes: list[str] = []
    sec = lambda name: u[name] if isinstance(u.get(name), dict) else None
    r = lambda x: float(f"{x:.4g}")

    if (b := sec("background")) and ({"darken", "contrast"} & b.keys()):
        b["brightness"] = r(b.pop("darken", 0.6) * b.pop("contrast", 1.1))
        notes.append("background.darken × contrast → brightness")

    for name in ("organic", "stretch"):
        s = sec(name)
        given = {k for k in (s or {}) if k in _OLD_WAVE}
        if not given:
            continue
        o = {**_OLD[name], **{k: s.pop(k) for k in given}}
        if "speed" in given:
            s["period"] = r(1 / (max(o["speed"], 1e-9) * 30))
        if "scale" in given:
            s["size"] = r(1 / max(o["scale"], 1e-9))
        if name == "organic" and "intensity" in given:
            s["amount"] = r(o["intensity"])
        if name == "stretch":
            if given & {"intensity", "horizontal"}:
                s["horizontal"] = r(o["intensity"] * 15 * o["horizontal"])
            if given & {"intensity", "vertical"}:
                s["vertical"] = r(o["intensity"] * 15 * o["vertical"])
        if given & {"audio_reactive", "bass_speed"}:
            s["audio"] = r(min(1.0, o["bass_speed"] / max(o["speed"], 1e-9))) if o["audio_reactive"] else 0.0
        notes.append(f"{name}: {', '.join(sorted(given))} → amount/size/period/audio")

    a = sec("audio")
    lens_audio = a.pop("bass", None) if a else None
    if a and ({"mid", "high"} & a.keys()):
        a.pop("mid", None), a.pop("high", None)
    if (l := sec("lenses")) and (_OLD_LENS & l.keys() or "speed" in l or l.get("motion") == "drift"):
        drift = l.get("motion") == "drift"
        if drift:
            l["speed"] = r(l.pop("drift_speed", 0.6) * 30)
        elif "speed" in l and l["speed"] < 1:  # i px/frame vecchi: valori minuscoli = ferme
            l["speed"] = r(l["speed"] * 30) if l["speed"] * 30 >= 0.05 else 0
        elif "speed" in l:
            l["speed"] = r(l["speed"] * 30)
        dropped = sorted(_OLD_LENS & l.keys())
        for k in dropped:
            l.pop(k)
        notes.append("lenses: velocità in px/s" + (f", tolte {', '.join(dropped)}" if dropped else ""))
    if lens_audio is not None:
        u.setdefault("lenses", {})["audio"] = r(min(1.0, lens_audio))
        notes.append("audio.bass → lenses.audio")

    for t in ("logo", "background"):
        tr = (sec("tracers") or {}).get(t)
        if isinstance(tr, dict) and ({"threshold1", "threshold2"} & tr.keys()):
            tr.pop("threshold1", None)
            tr["threshold"] = tr.pop("threshold2", 21)
            notes.append(f"tracers.{t}.threshold1/2 → threshold")

    if (bl := sec("blend")) and ({"transparency", "adaptive"} & bl.keys()):
        tr = bl.pop("transparency", 0.0)
        if tr:
            bl["strength"] = r(bl.get("strength", 0.5) * (1 - tr))
        if bl.pop("adaptive", True) is False:
            bl["harmonize"] = bl["luminance_match"] = False
        notes.append("blend: transparency dentro strength, adaptive tolto")

    if u.pop("variation", None) is not None:
        notes.append("variation tolta")
    return u, notes


class ConfigError(Exception):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def find_preset(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".toml", ".json") and p.exists():
        return p
    cand = PRESETS_DIR / f"{name}.toml"
    if cand.exists():
        return cand
    raise ConfigError(f"Preset '{name}' non trovato (cercato {p} e {cand}). "
                      f"Disponibili: {', '.join(list_presets()) or 'nessuno'}")


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.toml") if not p.name.startswith("._"))


def _load_file(path: Path) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("config", data)  # i .json salvati accanto ai video hanno {"config":..., "run":...}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_value(raw: str):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw  # stringa senza virgolette


def parse_override(expr: str) -> dict:
    if "=" not in expr:
        raise ConfigError(f"--set vuole chiave=valore, es. lenses.count=20 (ricevuto '{expr}')")
    key, raw = expr.split("=", 1)
    parts = key.strip().split(".")
    d: dict = {}
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = _parse_value(raw.strip())
    return d


def _check_keys(defaults: dict, user: dict, prefix=""):
    """Avvisa sulle chiavi sconosciute: un typo nel preset altrimenti non si nota."""
    for k, v in user.items():
        name = f"{prefix}{k}"
        if k not in defaults:
            raise ConfigError(f"Chiave sconosciuta '{name}'. Controlla crystal/defaults.toml per i nomi validi.")
        if isinstance(v, dict) and isinstance(defaults[k], dict):
            _check_keys(defaults[k], v, name + ".")


def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    return d


def load_defaults() -> dict:
    with open(DEFAULTS_FILE, "rb") as f:
        return tomllib.load(f)


def load_user(preset: str | None = None, overrides: list[str] = ()) -> tuple[dict, str, Path | None]:
    """Le chiavi scritte dall'utente (preset + --set), il nome del preset e il suo file."""
    user: dict = {}
    name, path = "default", None
    if preset:
        path = find_preset(preset)
        user, notes = migrate(_load_file(path))
        if notes and path.suffix == ".toml":
            print(f"ℹ️  {path.name} usa nomi vecchi dei parametri, convertiti al volo: " + "; ".join(notes))
        name = path.stem
        if path.suffix == ".json":  # video rifatto da un .json: tieni il nome del preset originale
            name = json.loads(path.read_text()).get("preset", path.stem)
    for o in overrides:
        user = deep_merge(user, parse_override(o))
    return user, name, path


def resolve(user: dict, test: bool = False, defaults: dict | None = None) -> dict:
    """defaults ← chiavi dell'utente (con i preset di blending) → config completo (dict)."""
    defaults = defaults if defaults is not None else load_defaults()
    _check_keys(defaults, user)
    data = deep_merge(defaults, user)

    # preset di blending: default < preset < chiavi esplicite dell'utente
    bp = data["blend"]["preset"]
    if bp != "manual":
        if bp not in BLEND_PRESETS:
            raise ConfigError(f"blend.preset '{bp}' sconosciuto. Validi: manual, {', '.join(BLEND_PRESETS)}")
        data["blend"] = {**defaults["blend"], **BLEND_PRESETS[bp], **user.get("blend", {})}

    if test:
        data["output"]["scale"] = data["test"]["scale"]
        data["output"]["fps"] = data["test"]["fps"]
        data["output"]["duration"] = min(data["output"]["duration"], data["test"]["duration"])
        if Path(data["output"]["dir"]).name != "test":
            data["output"]["dir"] = str(Path(data["output"]["dir"]) / "test")
    return data


def load_config(preset: str | None = None, overrides: list[str] = (), test: bool = False):
    """Ritorna (cfg_namespace, cfg_dict, nome_preset)."""
    user, name, _ = load_user(preset, overrides)
    data = resolve(user, test)
    return to_ns(data), data, name
