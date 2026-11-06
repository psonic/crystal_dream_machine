"""
Schema dei parametri per l'interfaccia dello studio, letto da defaults.toml.

Da defaults.toml prende: sezioni, chiavi, valori di default e i commenti (diventano
le descrizioni). Qui sotto ci sono solo le informazioni che il file non ha: range
degli slider, scelte possibili, tipi di file. Una chiave nuova in defaults.toml
compare da sola nello studio (con un range ricavato dal valore); se vuoi uno
slider migliore aggiungila a HINTS.
"""

from __future__ import annotations

import re
import tomllib

from ..config import BLEND_PRESETS, DEFAULTS_FILE
from ..effects.compose import BLEND_MODES

# key → dict(min, max, step, log) per i numeri, choices per le scelte, files per i file
HINTS: dict[str, dict] = {
    "output.format": {"choices": ["ig_post", "ig_story", "square", "landscape", "logo", "custom"]},
    "output.width": {"min": 100, "max": 4000, "step": 2},
    "output.height": {"min": 100, "max": 4000, "step": 2},
    "output.scale": {"min": 0.1, "max": 2, "step": 0.05},
    "output.fps": {"min": 10, "max": 60, "step": 1},
    "output.duration": {"min": 1, "max": 180, "step": 1},
    "output.crf": {"min": 0, "max": 35, "step": 1},
    "output.x264_preset": {"choices": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
                                       "slow", "slower", "veryslow"]},
    "output.encoder": {"choices": ["x264", "videotoolbox"]},
    "output.workers": {"min": 0, "max": 16, "step": 1},
    "test.scale": {"min": 0.1, "max": 1, "step": 0.05},
    "test.fps": {"min": 5, "max": 60, "step": 1},
    "test.duration": {"min": 1, "max": 60, "step": 1},

    "logo.file": {"files": "logo"},
    "logo.padding": {"min": 0, "max": 400, "step": 1},
    "logo.left_padding": {"min": -400, "max": 400, "step": 1},
    "logo.zoom": {"min": 0.2, "max": 3, "step": 0.01},
    "logo.fill": {"choices": ["blob", "none", "holes"]},

    "background.video": {"files": "video"},
    "background.fit": {"choices": ["cover", "crop"]},
    "background.zoom": {"min": 1, "max": 3, "step": 0.01},
    "background.slowdown": {"min": 0.25, "max": 4, "step": 0.05},
    "background.darken": {"min": 0, "max": 2, "step": 0.01},
    "background.contrast": {"min": 0, "max": 2, "step": 0.01},
    "background.start": {"min": 0, "max": 300, "step": 0.1},
    "background.crop_x": {"min": 0, "max": 1, "step": 0.01},
    "background.crop_y": {"min": 0, "max": 1, "step": 0.01},
    "background.crop_w": {"min": 0.05, "max": 1, "step": 0.01},
    "background.crop_h": {"min": 0.05, "max": 1, "step": 0.01},

    "texture.path": {"files": "image"},
    "texture.target": {"choices": ["logo", "background", "both"]},
    "texture.alpha": {"min": 0, "max": 1, "step": 0.01},
    "texture.background_alpha": {"min": 0, "max": 1, "step": 0.01},
    "texture.mode": {"choices": BLEND_MODES},

    "glow.size": {"min": 1, "max": 100, "step": 1},
    "glow.intensity": {"min": 0, "max": 2, "step": 0.01},

    "organic.speed": {"min": 0.0001, "max": 0.2, "log": True},
    "organic.scale": {"min": 0.0001, "max": 0.01, "log": True},
    "organic.intensity": {"min": 0, "max": 100, "step": 0.5},
    "organic.bass_intensity": {"min": 0, "max": 0.2, "step": 0.001},
    "organic.bass_speed": {"min": 0, "max": 0.02, "step": 0.0005},
    "organic.mid_scale": {"min": 0, "max": 0.002, "step": 0.00005},
    "organic.smoothing": {"min": 0, "max": 0.995, "step": 0.005},

    "stretch.speed": {"min": 0.00001, "max": 0.05, "log": True},
    "stretch.scale": {"min": 0.0001, "max": 0.01, "log": True},
    "stretch.intensity": {"min": 0, "max": 100, "step": 0.5},
    "stretch.horizontal": {"min": 0, "max": 2, "step": 0.01},
    "stretch.vertical": {"min": 0, "max": 2, "step": 0.01},
    "stretch.fine_detail": {"min": 0, "max": 2, "step": 0.01},
    "stretch.bass_intensity": {"min": 0, "max": 0.2, "step": 0.001},
    "stretch.bass_speed": {"min": 0, "max": 0.02, "step": 0.0005},
    "stretch.mid_scale": {"min": 0, "max": 0.002, "step": 0.00005},
    "stretch.smoothing": {"min": 0, "max": 0.995, "step": 0.005},

    "lenses.count": {"min": 0, "max": 300, "step": 1},
    "lenses.min_strength": {"min": -3, "max": 3, "step": 0.05},
    "lenses.max_strength": {"min": -3, "max": 3, "step": 0.05},
    "lenses.min_radius": {"min": 1, "max": 300, "step": 1},
    "lenses.max_radius": {"min": 1, "max": 400, "step": 1},
    "lenses.speed": {"min": 0.000001, "max": 5, "log": True},
    "lenses.path_speed": {"min": 0.0001, "max": 2, "log": True},
    "lenses.inertia": {"min": 0, "max": 0.99, "step": 0.01},
    "lenses.rotation_min": {"min": -0.1, "max": 0.1, "step": 0.001},
    "lenses.rotation_max": {"min": -0.1, "max": 0.1, "step": 0.001},
    "lenses.rotation_speed": {"min": 0.0001, "max": 5, "log": True},
    "lenses.pulsation_speed": {"min": 0.00001, "max": 0.2, "log": True},
    "lenses.pulsation_amplitude": {"min": 0, "max": 2, "step": 0.01},
    "lenses.force_pulsation_amplitude": {"min": 0, "max": 2, "step": 0.01},
    "lenses.worm_length": {"min": 0.2, "max": 5, "step": 0.05},
    "lenses.worm_wave": {"min": 0, "max": 150, "step": 1},

    "tracers.logo.trail": {"min": 1, "max": 300, "step": 1},
    "tracers.logo.opacity": {"min": 0, "max": 1, "step": 0.005},
    "tracers.logo.hue_shift": {"min": 0, "max": 10, "step": 0.05},
    "tracers.background.trail": {"min": 1, "max": 300, "step": 1},
    "tracers.background.opacity": {"min": 0, "max": 1, "step": 0.005},
    "tracers.background.hue_shift": {"min": 0, "max": 10, "step": 0.05},
    "tracers.background.threshold1": {"min": 0, "max": 255, "step": 1},
    "tracers.background.threshold2": {"min": 0, "max": 255, "step": 1},

    "blend.preset": {"choices": ["manual", *BLEND_PRESETS]},
    "blend.mode": {"choices": BLEND_MODES},
    "blend.strength": {"min": 0, "max": 1, "step": 0.01},
    "blend.edge_blur": {"min": 0, "max": 20, "step": 1},
    "blend.transparency": {"min": 0, "max": 1, "step": 0.01},
    "blend.color_strength": {"min": 0, "max": 1, "step": 0.01},
    "blend.edge_softness": {"min": 0, "max": 30, "step": 1},

    "audio.files": {"files": "audio"},
    "audio.start": {"min": 0, "max": 600, "step": 0.1},
    "audio.bass": {"min": 0, "max": 2, "step": 0.005},
    "audio.mid": {"min": 0, "max": 2, "step": 0.005},
    "audio.high": {"min": 0, "max": 2, "step": 0.005},
    "audio.smoothing": {"min": 0, "max": 0.995, "step": 0.005},

    "variation.amplitude": {"min": 0, "max": 2, "step": 0.01},
    "variation.speed_medium": {"min": 0, "max": 0.2, "step": 0.001},
    "variation.speed_fast": {"min": 0, "max": 0.2, "step": 0.001},
}

# titoli delle sezioni (in ordine di come conviene vederle nello studio)
TITLES = {
    "lenses": "Lenti",
    "organic": "Organic · ondulazioni",
    "stretch": "Stretch · stiramenti",
    "tracers.logo": "Scie del logo",
    "tracers.background": "Scie dello sfondo",
    "blend": "Fusione logo/sfondo",
    "glow": "Glow",
    "logo": "Logo",
    "background": "Sfondo",
    "texture": "Texture",
    "audio": "Audio",
    "variation": "Variazioni lente",
    "output": "Output (video finale)",
    "test": "Render di test",
}

# descrizioni per le chiavi che in defaults.toml non hanno un commento
HELP = {
    "lenses.enabled": "lenti che passano sul logo e lo deformano",
    "lenses.count": "quante lenti",
    "lenses.max_strength": "forza massima (le forze sono distribuite fra min e max)",
    "lenses.min_radius": "raggio minimo (px)",
    "lenses.rotation_min": "rotazione per frame: ogni lente ne prende una a caso fra min e max",
    "lenses.rotation_max": "rotazione per frame: ogni lente ne prende una a caso fra min e max",
    "lenses.rotation_speed": "moltiplicatore della rotazione",
    "lenses.pulsation": "il raggio delle lenti pulsa",
    "lenses.pulsation_speed": "velocità della pulsazione",
    "lenses.pulsation_amplitude": "quanto pulsa il raggio",
    "lenses.force_pulsation": "anche la forza pulsa",
    "lenses.force_pulsation_amplitude": "quanto pulsa la forza",
    "lenses.worm_length": "allungamento del verme (1 = tondo)",
    "organic.enabled": "ondulazioni morbide di tutto il logo",
    "organic.speed": "velocità con cui le onde cambiano",
    "organic.audio_reactive": "i bassi muovono le ondulazioni",
    "stretch.enabled": "stiramenti lunghi, soprattutto orizzontali",
    "stretch.speed": "velocità con cui gli stiramenti cambiano",
    "stretch.scale": "frequenza: più alto = stiramenti più corti",
    "stretch.intensity": "forza degli stiramenti",
    "stretch.audio_reactive": "i bassi muovono gli stiramenti",
    "tracers.logo.opacity": "luminosità della scia",
    "tracers.logo.color": "colore di partenza (la tinta poi ruota)",
    "tracers.background.enabled": "scie dei contorni del video di sfondo",
    "tracers.background.trail": "lunghezza in frame (a 30 fps)",
    "tracers.background.opacity": "luminosità della scia: su acqua/fumo tenerla bassa",
    "tracers.background.color": "colore di partenza (la tinta poi ruota)",
    "tracers.background.hue_shift": "cambio di tinta lungo la scia",
    "tracers.background.threshold2": "soglia alta di Canny",
    "blend.mode": "come si combinano i colori di logo e sfondo",
    "blend.strength": "forza della fusione",
    "blend.edge_detection": "bordi del logo fusi a parte",
    "blend.edge_blur": "sfocatura dei bordi (px)",
    "blend.harmonize": "il logo prende il colore medio dello sfondo",
    "blend.luminance_match": "il logo prende la luminosità dello sfondo",
    "blend.transparency": "quanto sfondo si vede attraverso il logo",
    "blend.color_strength": "quanto il logo prende il colore dello sfondo",
    "glow.enabled": "bagliore attorno al logo",
    "logo.color": "colore del logo",
    "logo.zoom": "ingrandimento del logo",
    "background.enabled": "false = sfondo nero",
    "background.crop_y": "solo fit = crop",
    "background.crop_w": "solo fit = crop",
    "background.crop_h": "solo fit = crop",
    "texture.enabled": "un'immagine fusa su logo e/o sfondo",
    "texture.path": "l'immagine",
    "audio.enabled": "reattività all'audio",
    "audio.random_start": "parte da un punto a caso della traccia",
    "variation.enabled": "oscillazioni lentissime automatiche di lenti e scie",
    "variation.amplitude": "ampiezza delle oscillazioni",
    "output.height": "usati solo con format = custom",
    "output.fps": "fotogrammi al secondo",
    "output.dir": "cartella dei video",
    "test.scale": "risoluzione del render di test",
    "test.fps": "fps del render di test",
    "test.duration": "durata massima del render di test (s)",
}

# chiavi che lo studio gestisce altrove (barra in alto)
HIDDEN = {"seed"}

_SECTION = re.compile(r"^\[([\w.]+)\]\s*(?:#\s*(.*))?$")
_KEY = re.compile(r"^(\w+)\s*=")


def split_comment(line: str) -> tuple[str, str]:
    """'a = "x#y"  # nota' → ('a = "x#y"', 'nota'), ignorando i # dentro le stringhe."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == "\\":
                continue
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:i].rstrip(), line[i + 1:].strip()
    return line.rstrip(), ""


def _number_hint(key, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return {}
    if isinstance(v, int):
        return {"min": min(0, v), "max": max(10, v * 4), "step": 1}
    a = abs(v) or 1.0
    return {"min": min(0.0, v * 4), "max": max(a * 4, 1.0 if a >= 0.5 else a * 4), "step": a / 100}


def build_schema() -> list[dict]:
    """Gruppi di parametri, ognuno con le sue chiavi, nell'ordine di TITLES."""
    text = DEFAULTS_FILE.read_text()
    with open(DEFAULTS_FILE, "rb") as f:
        defaults = tomllib.load(f)

    groups: dict[str, dict] = {}
    sec, last = "", None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            last = None
            continue
        m = _SECTION.match(line)
        if m:
            sec = m.group(1)
            groups[sec] = {"id": sec, "title": TITLES.get(sec, sec), "help": m.group(2) or "", "params": []}
            last = None
            continue
        if line.startswith("#"):
            # commento su una riga sua: continua la descrizione della chiave precedente
            # se è indentato, altrimenti è una nota della sezione
            note = line.lstrip("#").strip()
            if last is not None and raw.startswith((" ", "\t")):
                last["help"] = (last["help"] + " " + note).strip()
            elif sec in groups and note and "───" not in note:
                groups[sec]["help"] = (groups[sec]["help"] + " " + note).strip()
            continue
        m = _KEY.match(line)
        if not m:
            continue
        name = m.group(1)
        key = f"{sec}.{name}" if sec else name
        if key in HIDDEN:
            last = None
            continue
        node = defaults
        for part in (sec.split(".") if sec else []):
            node = node[part]
        value = node[name]
        _, comment = split_comment(line)
        p = {"key": key, "name": name, "default": value, "help": comment or HELP.get(key, "")}
        h = HINTS.get(key, {})
        if isinstance(value, bool):
            p["type"] = "bool"
        elif "choices" in h:
            p.update(type="choice", choices=h["choices"])
        elif "files" in h:
            p.update(type="file", files=h["files"])
        elif name == "color" and isinstance(value, list):
            p["type"] = "color"
        elif isinstance(value, (int, float)):
            p.update(type="int" if isinstance(value, int) else "float", **(h or _number_hint(key, value)))
            if h.get("log"):
                p["type"] = "float"
        else:
            p["type"] = "text"
        groups.setdefault(sec, {"id": sec, "title": TITLES.get(sec, sec), "help": "", "params": []})
        groups[sec]["params"].append(p)
        last = p

    order = [g for g in TITLES if g in groups] + [g for g in groups if g not in TITLES]
    return [groups[g] for g in order if groups[g]["params"]]
