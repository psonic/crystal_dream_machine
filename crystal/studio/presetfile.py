"""
Scrittura dei preset .toml dallo studio.

Il preset contiene solo le chiavi diverse dai default. Quando lo studio salva,
riscrive il file mantenendo i commenti che c'erano: quelli in testa al file,
quelli sopra una sezione o una chiave e quelli a fine riga.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path

from .schema import split_comment

_SECTION = re.compile(r"^\[([\w.]+)\]")
_KEY = re.compile(r"^(\w+)\s*=")


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"valore non valido: {v}")
        r = repr(v)
        if "e" in r or "E" in r:
            mant, exp = r.lower().split("e")
            if "." not in mant:
                mant += ".0"
            r = f"{mant}e{int(exp)}"
        return r
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise ValueError(f"tipo non supportato nel preset: {type(v).__name__}")


def flatten(user: dict, prefix: str = "") -> dict[str, dict]:
    """{"tracers": {"logo": {"opacity": 1}}} → {"tracers.logo": {"opacity": 1}}; "" = chiavi in cima."""
    out: dict[str, dict] = {}
    for k, v in user.items():
        if isinstance(v, dict):
            for sec, kv in flatten(v, f"{prefix}{k}.").items():
                out.setdefault(sec, {}).update(kv)
        else:
            out.setdefault(prefix.rstrip("."), {})[k] = v
    return out


def _parse_existing(text: str):
    """Commenti del file esistente: testa, sopra/accanto a sezioni e chiavi, ordine delle chiavi."""
    header, pending = [], []
    above: dict[tuple, list[str]] = {}
    inline: dict[tuple, str] = {}
    raw_line: dict[tuple, str] = {}
    order: list[tuple] = []
    sec, seen_item = "", False
    for raw in text.splitlines():
        line = raw.strip()
        if not seen_item and (not line or line.startswith("#")):
            header.append(raw)
            continue
        if not line:
            pending.append("")
            continue
        if line.startswith("#"):
            pending.append(raw)
            continue
        seen_item = True
        m = _SECTION.match(line)
        if m:
            sec = m.group(1)
            ident = ("[", sec)
        else:
            m = _KEY.match(line)
            if not m:
                pending = []
                continue
            ident = (sec, m.group(1))
        while pending and pending[0] == "":
            pending.pop(0)
        above[ident] = [p for p in pending if p != ""] if ident[0] != "[" else pending
        _, c = split_comment(line)
        if c:
            inline[ident] = c
        raw_line[ident] = raw.rstrip()
        order.append(ident)
        pending = []
    # le righe vuote finali della testa vanno via; i commenti in fondo al file restano
    while header and not header[-1].strip():
        header.pop()
    return header, above, inline, raw_line, order, pending


def render_preset(user: dict, defaults: dict, old_text: str = "", title: str = "") -> str:
    header, above, inline, raw_line, order, tail = _parse_existing(old_text)
    flat = flatten(user)
    try:
        old_flat = flatten(tomllib.loads(old_text))
    except tomllib.TOMLDecodeError:
        old_flat = {}

    def def_order(sec):
        node = defaults
        for part in (sec.split(".") if sec else []):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        return [k for k, v in node.items() if not isinstance(v, dict)] if isinstance(node, dict) else []

    def all_sections(d, prefix=""):
        out = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.append(prefix + k)
                out += all_sections(v, prefix + k + ".")
        return out

    old_secs = [i[1] for i in order if i[0] == "["]
    secs = [s for s in old_secs if s in flat] + [s for s in all_sections(defaults) if s in flat and s not in old_secs]
    secs += [s for s in flat if s and s not in secs]

    lines: list[str] = []
    if header:
        lines += header
    elif title:
        lines.append(f"# {title}")

    def emit_keys(sec):
        kv = flat.get(sec, {})
        old_keys = [i[1] for i in order if i[0] == sec]
        keys = [k for k in old_keys if k in kv] + [k for k in def_order(sec) if k in kv and k not in old_keys]
        keys += [k for k in kv if k not in keys]
        width = max((len(k) for k in keys), default=0)
        for k in keys:
            lines.extend(above.get((sec, k), []))
            if (sec, k) in raw_line and old_flat.get(sec, {}).get(k, object()) == kv[k]:
                lines.append(raw_line[(sec, k)])  # invariata: la riga resta com'era
                continue
            text = f"{k.ljust(width)} = {toml_value(kv[k])}"
            c = inline.get((sec, k))
            lines.append(f"{text}   # {c}" if c else text)

    if flat.get(""):
        lines.append("")
        emit_keys("")
    for sec in secs:
        lines.append("")
        lines.extend(c for c in above.get(("[", sec), []) if c != "")
        c = inline.get(("[", sec))
        lines.append(raw_line.get(("[", sec)) or f"[{sec}]" + (f"   # {c}" if c else ""))
        emit_keys(sec)
    tail = [t for t in tail if t.strip()]
    if tail:
        lines.append("")
        lines += tail
    text = "\n".join(lines).lstrip("\n") + "\n"
    tomllib.loads(text)  # sicurezza: deve essere un TOML valido
    return text


def write_preset(path: Path, user: dict, defaults: dict, title: str = ""):
    old = path.read_text() if path.exists() else ""
    text = render_preset(user, defaults, old, title)
    if tomllib.loads(text) != user:
        raise ValueError("il preset riletto non coincide con quello da salvare")
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(text)
    tmp.replace(path)
