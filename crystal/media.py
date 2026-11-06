"""
I/O multimediale via ffmpeg: logo SVG → maschera, lettura video di sfondo, scrittura video, audio.

ffmpeg fa scaling/crop/rallentamento dello sfondo e l'encoding x264 nel suo processo,
mentre Python calcola i frame: le due cose girano in parallelo.
"""

from __future__ import annotations

import functools
import glob
import io
import os
import queue
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  File e probe
# ═════════════════════════════════════════════════════════════════════════════

def _mtime(path) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def cached(maxsize=16):
    """
    Cache in memoria per funzioni lente con un file come primo argomento (logo, audio, probe).
    La chiave include la data di modifica del file: se lo salvi di nuovo, viene ricaricato.
    Gli array restituiti sono in sola lettura (sono condivisi fra chi li chiede).
    """
    def deco(fn):
        @functools.lru_cache(maxsize=maxsize)
        def inner(mtime, *args):
            out = fn(*args)
            if isinstance(out, np.ndarray):
                out.flags.writeable = False
            return out

        @functools.wraps(fn)
        def wrapper(path, *args):
            return inner(_mtime(path), path, *args)
        return wrapper
    return deco


def expand_files(spec) -> list[str]:
    """Stringa, glob o lista → lista di file esistenti."""
    items = [spec] if isinstance(spec, str) else list(spec or [])
    out = []
    for it in items:
        matches = sorted(glob.glob(it))
        out.extend(matches if matches else ([it] if Path(it).exists() else []))
    return out


@cached(64)
def probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Logo SVG → maschera
# ═════════════════════════════════════════════════════════════════════════════

def svg_size(path: str) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, w, h = map(float, vb.replace(",", " ").split())
        return int(round(w)), int(round(h))
    num = lambda v: float("".join(c for c in v if c.isdigit() or c == "."))
    return int(num(root.get("width", "1920"))), int(num(root.get("height", "1080")))


@cached()
def logo_size(path: str) -> tuple[int, int]:
    if Path(path).suffix.lower() == ".pdf":
        import pymupdf
        r = pymupdf.open(path)[0].rect
        return int(round(r.width)), int(round(r.height))
    return svg_size(path)


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def svg_polylines(svg_path: str) -> list[np.ndarray]:
    """Ogni path SVG campionato come UNA polilinea, come faceva la v3 (anche i sotto-path vengono uniti)."""
    from svgpathtools import svg2paths2
    paths, _, _ = svg2paths2(svg_path)
    polys = []
    for p in paths:
        length = p.length()
        if length == 0:
            continue
        n = max(100, min(1000, int(length * 3)))
        pts = [p.point(t) for t in np.linspace(0, 1, n)]
        arr = np.array([[z.real, z.imag] for z in pts if not np.isnan(z.real)], np.float64)
        if len(arr) > 10:
            polys.append(arr)
    return polys


def blob_mask(svg_path: str, W, H, padding, left_padding, zoom, s) -> np.ndarray:
    """
    Il look della v3: i path riempiti come poligoni → zone intere piene, poi sagoma
    più grande con i buchi chiusi. Era un effetto collaterale, ma è il look dei video.
    """
    polys = svg_polylines(svg_path)
    pts = np.vstack(polys)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    bw, bh = mx - mn
    scale = min((W - 2 * padding) / bw, (H - 2 * padding) / bh) * zoom
    off = np.array([(W - bw * scale) / 2 + left_padding, (H - bh * scale) / 2])
    ss = 4  # supersampling per bordi lisci
    big = np.zeros((H * ss, W * ss), np.uint8)
    for p in polys:
        q = ((p - mn) * scale + off) * ss
        cv2.fillPoly(big, [np.round(q).astype(np.int32)], 255)
    mask = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return blobify(mask, s)


def blobify(mask: np.ndarray, s: float) -> np.ndarray:
    """Unisce i pezzi vicini, tiene la sagoma più grande e ne riempie i buchi (create_unified_mask della v3)."""
    b = np.where(mask > 127, 255, 0).astype(np.uint8)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(9 * s),) * 2)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k9, iterations=2)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(b)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1, lineType=cv2.LINE_AA)
    k = _odd(5 * s)
    out = np.where(cv2.GaussianBlur(out, (k, k), 0) > 127, 255, 0).astype(np.uint8)  # bordo liscio
    return cv2.GaussianBlur(out, (3, 3), 0)  # antialiasing


def _raster_svg(path: str, render_w: int) -> np.ndarray:
    import cairosvg
    from PIL import Image
    sw, sh = svg_size(path)
    png = cairosvg.svg2png(url=path, output_width=render_w, output_height=int(render_w * sh / sw))
    rgba = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    alpha = rgba[:, :, 3]
    if alpha.max() < 5:  # SVG senza trasparenza: usa la luminosità invertita
        alpha = 255 - cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    return alpha


def _raster_pdf(path: str, render_w: int) -> np.ndarray:
    """Prima pagina del PDF. Logo = pixel scuri (come la v3: soglia a 50 su sfondo bianco)."""
    try:
        import pymupdf
    except ImportError:
        raise RuntimeError("Per i loghi PDF serve PyMuPDF:  .venv/bin/python -m pip install pymupdf")
    page = pymupdf.open(path)[0]
    ss = 2  # supersampling: la soglia netta poi viene ammorbidita dal ridimensionamento
    z = render_w * ss / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    hard = np.where(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) <= 50, 255, 0).astype(np.uint8)
    return cv2.resize(hard, (pix.width // ss, pix.height // ss), interpolation=cv2.INTER_AREA)


@cached()
def load_logo_mask(path: str, W: int, H: int, padding: float, left_padding: float,
                   zoom: float, fill: str, s: float) -> np.ndarray:
    """Rasterizza il logo (SVG o PDF), lo ritaglia al contenuto e lo centra nel canvas. uint8 (H,W) 0-255."""
    is_pdf = Path(path).suffix.lower() == ".pdf"
    if fill == "blob" and not is_pdf:
        return blob_mask(path, W, H, padding, left_padding, zoom, s)

    render_w = max(2000, int(W * zoom * 2))
    alpha = _raster_pdf(path, render_w) if is_pdf else _raster_svg(path, render_w)

    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        raise RuntimeError(f"Il logo {path} sembra vuoto")
    crop = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    if fill == "holes":
        cnts, _ = cv2.findContours((crop > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(crop)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        crop = np.maximum(crop, filled)
    elif fill not in ("none", "blob"):
        raise ValueError(f"logo.fill '{fill}' non valido: none | holes | blob")

    ch, cw = crop.shape
    scale = min((W - 2 * padding) / cw, (H - 2 * padding) / ch) * zoom
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    logo = cv2.resize(crop, (nw, nh), interpolation=interp)

    # incolla centrato (con clipping se lo zoom lo fa uscire dal frame)
    mask = np.zeros((H, W), np.uint8)
    ox = int(round((W - nw) / 2 + left_padding))
    oy = int(round((H - nh) / 2))
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + nw), min(H, oy + nh)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = logo[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return blobify(mask, s) if fill == "blob" else mask  # (blob su PDF: sagoma piena)


# ═════════════════════════════════════════════════════════════════════════════
#  Thread di lettura/scrittura (così ffmpeg lavora mentre Python calcola)
# ═════════════════════════════════════════════════════════════════════════════

class _Prefetch:
    def __init__(self, fn, size=8):
        self.q: queue.Queue = queue.Queue(maxsize=size)
        self._stop = False
        self.t = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.t.start()

    def _run(self, fn):
        while not self._stop:
            item = fn()
            self.q.put(item)
            if item is None:
                return

    def get(self):
        return self.q.get()

    def stop(self):
        self._stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def background_filters(W: int, H: int, fps: float, cfg) -> str:
    """Catena di filtri ffmpeg: adatta lo sfondo al formato (cover/crop + zoom), rallenta, porta agli fps."""
    if cfg.fit == "crop":
        vf = [f"crop=iw*{cfg.crop_w}:ih*{cfg.crop_h}:(iw-iw*{cfg.crop_w})*{cfg.crop_x}:(ih-ih*{cfg.crop_h})*{cfg.crop_y}",
              f"scale={W}:{H}"]
    else:  # cover: riempie il frame mantenendo le proporzioni, poi zoom e crop centrale
        z = cfg.zoom
        r = W / H
        vf = [f"scale=w='if(gt(a,{r}),-2,{W * z})':h='if(gt(a,{r}),{H * z},-2)'",
              f"crop='min(iw,{W})':'min(ih,{H})'",
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"]
    vf += [f"setpts={cfg.slowdown}*PTS", f"fps={fps}"]
    return ",".join(vf)


# decodifica hardware (VideoToolbox) sul Mac: l'HEVC 4K dell'iPhone si decodifica ~2.5× più veloce
_HWACCEL = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []


class BackgroundReader:
    """Legge il video di sfondo già scalato/ritagliato/rallentato da ffmpeg, a partire da `start` secondi."""

    def __init__(self, path: str, W: int, H: int, fps: float, start: float, cfg, count: int | None = None):
        self.W, self.H = W, H
        self.frame_bytes = W * H * 3
        self.args = (path, background_filters(W, H, fps, cfg), start, count)
        self.got_any = False
        self.proc = self._spawn(_HWACCEL)
        self.pre = _Prefetch(self._read_one)

    def _spawn(self, hw):
        path, vf, start, count = self.args
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", *hw,
               "-ss", f"{start:.3f}", "-stream_loop", "-1", "-i", path,
               "-vf", vf, "-an"]
        if count:
            cmd += ["-frames:v", str(int(count))]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                bufsize=self.frame_bytes * 2)

    def _read_one(self):
        buf = self.proc.stdout.read(self.frame_bytes)
        if (not buf or len(buf) < self.frame_bytes) and not self.got_any and _HWACCEL:
            # la decodifica hardware non va con questo file: riprova in software
            self.proc.kill()
            self.proc.wait()
            self.proc = self._spawn([])
            buf = self.proc.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return None
        self.got_any = True
        return np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)

    def read(self) -> np.ndarray | None:
        return self.pre.get()

    def close(self):
        self.pre.stop()
        self.proc.kill()
        self.proc.wait()


class VideoWriter:
    """Encoding H.264 (yuv420p, faststart: va bene per WhatsApp/Instagram) + audio opzionale."""

    def __init__(self, path: str, W: int, H: int, fps: float, crf: int, preset: str,
                 audio: tuple[str, float, float] | None = None, encoder: str = "x264"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
        if audio:
            file, start, dur = audio
            cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", file,
                    "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        if encoder == "videotoolbox":
            # encoder hardware del Mac: quasi niente CPU. Qualità a bitrate costante, alta
            mbps = max(4, round(W * H * fps / 1e6 * 0.28 * (2 ** ((18 - crf) / 6))))
            cmd += ["-c:v", "h264_videotoolbox", "-b:v", f"{mbps}M", "-maxrate", f"{mbps * 2}M",
                    "-bufsize", f"{mbps * 2}M", "-profile:v", "high"]
        elif encoder == "x264":
            cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        else:
            raise ValueError(f"output.encoder '{encoder}' non valido: x264 | videotoolbox")
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError as e:
                self.error = e
                break

    def write(self, frame: np.ndarray):
        if self.error:
            raise RuntimeError("ffmpeg si è chiuso: " + self.proc.stderr.read().decode()[-800:])
        self.q.put(frame)

    def close(self):
        self.q.put(None)
        self.t.join()
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        err = self.proc.stderr.read().decode()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg ha fallito:\n{err[-800:]}")

    def abort(self):
        self.proc.kill()
        self.proc.wait()


# ═════════════════════════════════════════════════════════════════════════════
#  Audio
# ═════════════════════════════════════════════════════════════════════════════

@cached(8)
def decode_audio(path: str, start: float, duration: float, sr: int = 22050) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def audio_bands(y: np.ndarray, sr: int, fps: float, n_frames: int, n_fft: int = 2048) -> dict:
    """Energia di bassi (<250Hz), medi (250-4000Hz), alti (>4kHz) per frame, normalizzata 0-1."""
    half = n_fft // 2
    y = np.pad(y, (half, half + n_fft))
    hop = sr / fps
    starts = np.minimum((np.arange(n_frames) * hop).astype(np.int64), len(y) - n_fft)
    idx = starts[:, None] + np.arange(n_fft)[None, :]
    mag = np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft).astype(np.float32), axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    bands = {
        "bass": mag[:, freqs <= 250].mean(axis=1),
        "mid": mag[:, (freqs > 250) & (freqs <= 4000)].mean(axis=1),
        "high": mag[:, freqs > 4000].mean(axis=1),
    }
    return {k: (v / v.max() if v.max() > 0 else v).astype(np.float32) for k, v in bands.items()}
