"""
Riga di comando.

    python -m crystal render  [preset] [--test] [--set chiave=valore ...] [-o file.mp4]
    python -m crystal frame   [preset] [--at SECONDI] [--test] [-o file.png]
    python -m crystal preview [preset] [--set ...]
    python -m crystal presets
"""

import argparse
import sys
import time
from pathlib import Path

from .config import ConfigError, find_preset, list_presets, load_config


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
    warm = int(max(r.cfg.tracers.logo.trail, r.cfg.tracers.background.trail) * r.fps / 30) if args.trails else 0
    print(r.describe())
    print(f"  Frame    : {i} ({args.at}s){f', scaldo le scie con {warm} frame' if warm else ''}")
    frame = None
    for _, frame in r.frames(start=i, stop=i + 1, warmup=warm):
        pass
    out = Path(args.output or Path(r.cfg.output.dir) / f"frame_{r.preset_name}_{args.at:g}s.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"✅ {out}")


def cmd_preview(args):
    import cv2
    print("👁  Preview — q/esc esci · spazio pausa · s salva frame · r ricomincia · n nuovo seed")
    print("   Salva il file del preset e la preview si ricarica da sola.\n")
    preset_file = find_preset(args.preset) if args.preset else None
    seed_override = args.seed

    def mtime():
        return preset_file.stat().st_mtime if preset_file else 0

    while True:
        try:
            args.seed = seed_override
            r = _build(args, test=True)
        except (ConfigError, Exception) as e:  # errore nel preset: aspetta che venga corretto
            print(f"❌ {e}")
            m = mtime()
            while mtime() == m:
                time.sleep(0.5)
            continue
        print(r.describe())
        stamp, action = mtime(), None
        paused = False
        t_last = time.perf_counter()
        frames = r.frames()
        while action is None:
            if not paused:
                try:
                    i, frame = next(frames)
                except StopIteration:
                    action = "restart"
                    break
                now = time.perf_counter()
                cv2.imshow("crystal preview", frame)
                print(f"\r  frame {i + 1}/{r.n_frames}  {1 / max(now - t_last, 1e-6):4.1f} fps   ", end="", flush=True)
                t_last = now
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                frames.close()
                print()
                return
            if key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                out = Path(r.cfg.output.dir) / f"preview_{r.preset_name}_{i}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), frame)
                print(f"\n  💾 {out}")
            elif key == ord("r"):
                action = "restart"
            elif key == ord("n"):
                seed_override = 0
                action = "restart"
            if mtime() != stamp:
                print("\n  🔄 preset modificato, ricarico...")
                action = "reload"
        frames.close()
        if action == "restart" and seed_override is None:
            seed_override = r.seed  # "r" = stesso video da capo
        print()


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

    p = sub.add_parser("preview", help="anteprima live in una finestra, si ricarica quando salvi il preset")
    _common(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("presets", help="elenca i preset")
    p.set_defaults(func=cmd_presets)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as e:
        sys.exit(f"❌ {e}")
    except KeyboardInterrupt:
        sys.exit("\n⏹  interrotto")
