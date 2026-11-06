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
    "cinematic":   dict(mode="overlay",     strength=0.7,  edge_detection=False, edge_blur=3, adaptive=True,  harmonize=True,  luminance_match=False, transparency=0.2,  color_strength=0.9),
    "artistic":    dict(mode="difference",  strength=0.9,  edge_detection=True,  edge_blur=5, adaptive=False, harmonize=False, luminance_match=False, transparency=0.1,  color_strength=0.2),
    "soft":        dict(mode="soft_light",  strength=0.6,  edge_detection=False, edge_blur=3, adaptive=True,  harmonize=True,  luminance_match=True,  transparency=0.4,  color_strength=0.5),
    "dramatic":    dict(mode="multiply",    strength=0.85, edge_detection=True,  edge_blur=3, adaptive=True,  harmonize=False, luminance_match=True,  transparency=0.15, color_strength=0.3),
    "bright":      dict(mode="screen",      strength=0.7,  edge_detection=False, edge_blur=5, adaptive=True,  harmonize=True,  luminance_match=False, transparency=0.3,  color_strength=0.4),
    "intense":     dict(mode="hard_light",  strength=0.9,  edge_detection=False, edge_blur=3, adaptive=False, harmonize=False, luminance_match=False, transparency=0.1,  color_strength=0.25),
    "psychedelic": dict(mode="exclusion",   strength=0.95, edge_detection=True,  edge_blur=3, adaptive=False, harmonize=False, luminance_match=False, transparency=0.05, color_strength=0.1),
    "glow":        dict(mode="color_dodge", strength=0.75, edge_detection=False, edge_blur=3, adaptive=True,  harmonize=True,  luminance_match=False, transparency=0.2,  color_strength=0.35),
    "dark":        dict(mode="color_burn",  strength=0.8,  edge_detection=True,  edge_blur=3, adaptive=True,  harmonize=False, luminance_match=True,  transparency=0.25, color_strength=0.4),
    "geometric":   dict(mode="normal",      strength=1.0,  edge_detection=True,  edge_blur=1, adaptive=False, harmonize=False, luminance_match=False, transparency=0.0,  color_strength=0.0),
}


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
        user = _load_file(path)
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
