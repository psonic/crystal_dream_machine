"""
Riga di comando.

    python -m crystal render  [preset] [--test] [--set chiave=valore ...] [-o file.mp4]
    python -m crystal frame   [preset] [--at SECONDI] [--test] [-o file.png]
    python -m crystal studio  [preset] [--set ...]       (preview live nel browser; alias: preview)
    python -m crystal presets
"""

import argparse
import sys
from pathlib import Path

from .config import ConfigError, list_presets, load_config


def _common(p):
    p.add_argument("preset", nargs="?", help="nome in presets/ (senza .toml), un .toml o un .json salvato accanto a un video")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="CHIAVE=VALORE",
                   help="sovrascrive un parametro, es. --set lenses.count=20 (ripetibile)")
    p.add_argument("--seed", type=int, help="seed per le scelte casuali (stesso seed = stesso video)")
    p.add_argument("--test", action="store_true", help="veloce: risoluzione ridotta, meno fps, pochi secondi")


def _build(args, test=None):
    from .render import Renderer
    overrides = list(args.overrides) + ([f"seed={args.seed}"] if args.seed is not None else [])
    cfg, data, name = load_config(args.preset, overrides, args.test if test is None else test)
    return Renderer(cfg, data, name)


def cmd_render(args):
    r = _build(args)
    print("\n🎬 CRYSTAL — render" + ("  [TEST]" if args.test else ""))
    print(r.describe())
    r.render_video(args.output)


def cmd_frame(args):
    import cv2
    r = _build(args)
    i = min(r.n_frames - 1, int(args.at * r.fps))
    tr = r.cfg.tracers
    trail = max(tr.logo.trail if tr.logo.enabled else 0, tr.background.trail if tr.background.enabled else 0)
    warm = int(1.5 * trail * r.fps / 30) if args.trails else 0  # come lo studio: la scia scende sotto il 5%
    print(r.describe())
    print(f"  Frame    : {i} ({args.at}s){f', scaldo le scie con {warm} frame' if warm else ''}")
    frame = None
    for _, frame in r.frames(start=i, stop=i + 1, warmup=warm):
        pass
    out = Path(args.output or Path(r.cfg.output.dir) / f"frame_{r.preset_name}_{args.at:g}s.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"✅ {out}")


def cmd_studio(args):
    from .studio.server import serve
    serve(args.preset, args.overrides, args.seed, port=args.port, open_browser=not args.no_browser)


def cmd_presets(_args):
    for name in list_presets():
        print(" •", name)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m crystal", description="Crystal — video di logo deformati e psichedelici")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="genera un video")
    _common(p)
    p.add_argument("-o", "--output", help="file di output (default: output/<data>_<preset>.mp4)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("frame", help="salva un singolo frame PNG (per provare i parametri)")
    _common(p)
    p.add_argument("--at", type=float, default=3.0, help="secondo del video (default 3)")
    p.add_argument("--no-trails", dest="trails", action="store_false", help="non scaldare le scie (più veloce)")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_frame)

    for cmd in ("studio", "preview"):
        p = sub.add_parser(cmd, help="preview live nel browser con tutti i parametri" if cmd == "studio"
                           else "come studio")
        _common(p)
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true", help="non aprire il browser")
        p.set_defaults(func=cmd_studio)

    p = sub.add_parser("presets", help="elenca i preset")
    p.set_defaults(func=cmd_presets)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as e:
        sys.exit(f"❌ {e}")
    except KeyboardInterrupt:
        sys.exit("\n⏹  interrotto")
