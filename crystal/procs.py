"""
Processi di Crystal: chi gira, quanto consuma, come spegnerlo senza lasciare avanzi.

- I worker dei pool (deformazioni) controllano ogni secondo che il padre sia vivo:
  se il processo principale muore in qualsiasi modo (kill, crash, terminale chiuso)
  escono da soli invece di restare appesi per sempre.
- SIGTERM / SIGHUP nel processo principale diventano un KeyboardInterrupt, così i
  blocchi `finally` (chiusura dei pool, file temporanei) girano sempre.
- Ogni studio/render scrive un file in .cache/run/ (porta, preset, avanzamento):
  il pannello di controllo legge da lì.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from pathlib import Path

from .config import ROOT_DIR

RUN_DIR = ROOT_DIR / ".cache" / "run"


# ═════════════════════════════════════════════════════════════════════════════
#  Dentro i processi
# ═════════════════════════════════════════════════════════════════════════════
def watch_parent(interval: float = 1.0):
    """Nei worker: esce appena il processo padre non c'è più (adottati da launchd)."""
    parent = os.getppid()

    def loop():
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=loop, daemon=True, name="watch-parent").start()


def graceful_signals():
    """Nel processo principale: SIGTERM/SIGHUP si comportano come ctrl-c."""
    def handler(signum, _frame):
        signal.signal(signum, signal.SIG_IGN)  # un solo arresto, non a raffica
        raise KeyboardInterrupt

    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(s, handler)
        except (ValueError, OSError):  # non nel thread principale
            pass


def kill_pool(ex, timeout: float = 2.0):
    """Chiude un ProcessPoolExecutor senza aspettare il lavoro in corso, e senza lasciare processi."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        if p.is_alive():
            p.terminate()
    end = time.time() + timeout
    for p in procs:
        p.join(max(0.0, end - time.time()))
        if p.is_alive():
            p.kill()


# ═════════════════════════════════════════════════════════════════════════════
#  Registro: .cache/run/<tipo>_<pid>.json
# ═════════════════════════════════════════════════════════════════════════════
class Registration:
    def __init__(self, kind: str, **info):
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RUN_DIR / f"{kind}_{os.getpid()}.json"
        self.info = {"kind": kind, "pid": os.getpid(), "started": time.time(), **info}
        if os.environ.get("CRYSTAL_LOG"):  # avviato dal pannello: il suo log
            self.info["log"] = os.environ["CRYSTAL_LOG"]
        self._last = 0.0
        self.write()
        atexit.register(self.remove)

    def update(self, throttle: float = 0.0, **info):
        self.info.update(info)
        if throttle and time.time() - self._last < throttle:
            return
        self.write()

    def write(self):
        self._last = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.info, ensure_ascii=False, default=str))
        os.replace(tmp, self.path)

    def remove(self):
        self.path.unlink(missing_ok=True)


def registered() -> dict[int, dict]:
    """pid → info, solo per i processi ancora vivi (i file orfani vengono cancellati)."""
    out = {}
    if not RUN_DIR.exists():
        return out
    for f in RUN_DIR.glob("*_*.json"):
        try:
            info = json.loads(f.read_text())
            pid = int(info["pid"])
        except (OSError, ValueError, KeyError):
            continue
        if _alive(pid):
            out[pid] = info
        else:
            f.unlink(missing_ok=True)
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  Censimento (psutil)
# ═════════════════════════════════════════════════════════════════════════════
def _kind(cmd: list[str]) -> str | None:
    """Che tipo di processo Crystal è, dalla riga di comando."""
    if "-m" in cmd:
        i = cmd.index("-m")
        if i + 1 < len(cmd) and cmd[i + 1] == "crystal":
            sub = cmd[i + 2] if i + 2 < len(cmd) else ""
            return {"preview": "studio"}.get(sub, sub or "crystal")
    joined = " ".join(cmd)
    if "multiprocessing" in joined and "spawn_main" in joined:
        return "worker"
    if "multiprocessing" in joined and "resource_tracker" in joined:
        return "tracker"
    return None


class Census:
    """Tiene in vita gli oggetti psutil fra una lettura e l'altra: serve per la % di CPU."""

    def __init__(self):
        import psutil
        self.psutil = psutil
        self._procs: dict[int, object] = {}
        self.root = str(ROOT_DIR.resolve())

    def _proc(self, pid):
        p = self._procs.get(pid)
        if p is None:
            p = self._procs[pid] = self.psutil.Process(pid)
            p.cpu_percent(None)  # la prima lettura è sempre 0: fa da riferimento
        return p

    def scan(self) -> dict:
        ps = self.psutil
        me = os.getpid()
        rows = {}
        for p in ps.process_iter(["pid", "ppid", "cmdline", "create_time"]):
            try:
                cmd = p.info["cmdline"] or []
                kind = _kind(cmd)
                if not kind or p.info["pid"] == me:
                    continue
                if p.cwd() != self.root:  # solo i processi di questo progetto
                    continue
                q = self._proc(p.info["pid"])
                rows[p.info["pid"]] = {
                    "pid": p.info["pid"], "ppid": p.info["ppid"], "kind": kind,
                    "cmd": cmd, "started": p.info["create_time"],
                    "cpu": q.cpu_percent(None), "mem": q.memory_info().rss,
                }
            except (ps.NoSuchProcess, ps.AccessDenied, ps.ZombieProcess):
                continue
        for pid in [pid for pid in self._procs if pid not in rows]:
            del self._procs[pid]

        # anche i figli non-Python (ffmpeg) contano nel consumo di un render
        for pid in [pid for pid, r in rows.items() if r["kind"] != "worker"]:
            try:
                for c in self._proc(pid).children(recursive=True):
                    if c.pid not in rows:
                        cq = self._proc(c.pid)
                        rows[c.pid] = {"pid": c.pid, "ppid": cq.ppid(), "kind": "helper",
                                       "cmd": cq.cmdline()[:1], "started": cq.create_time(),
                                       "cpu": cq.cpu_percent(None), "mem": cq.memory_info().rss}
            except (ps.NoSuchProcess, ps.AccessDenied, ps.ZombieProcess):
                continue

        reg = registered()
        mains, orphans = [], []
        for r in rows.values():
            if r["kind"] in ("worker", "tracker", "helper"):
                continue
            kids = [k for k in rows.values() if k["pid"] != r["pid"] and self._descends(k, r["pid"], rows)]
            mains.append({
                **r, "info": reg.get(r["pid"], {}),
                "workers": len([k for k in kids if k["kind"] == "worker"]),
                "cpu_total": r["cpu"] + sum(k["cpu"] for k in kids),
                "mem_total": r["mem"] + sum(k["mem"] for k in kids),
                "children": [k["pid"] for k in kids],
            })
        for r in rows.values():
            if r["kind"] in ("worker", "tracker") and r["ppid"] == 1 and not any(r["pid"] in m["children"] for m in mains):
                orphans.append(r)
        mains.sort(key=lambda m: m["started"])
        return {
            "processes": mains,
            "orphans": orphans,
            "cpu": sum(r["cpu"] for r in rows.values()),
            "mem": sum(r["mem"] for r in rows.values()),
            "system": {"cpu": ps.cpu_percent(None), "cores": os.cpu_count(),
                       "load": os.getloadavg(), "mem": ps.virtual_memory().percent},
        }

    @staticmethod
    def _descends(row, ancestor, rows):
        seen = set()
        while row and row["ppid"] not in seen:
            if row["ppid"] == ancestor:
                return True
            seen.add(row["ppid"])
            row = rows.get(row["ppid"])
        return False


def stop(pid: int, timeout: float = 4.0) -> bool:
    """SIGTERM al processo (che chiude i suoi worker), poi SIGKILL a tutto l'albero se serve."""
    import psutil
    try:
        p = psutil.Process(pid)
        tree = [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return True
    p.terminate()
    _, alive = psutil.wait_procs([p], timeout=timeout)
    if alive:
        p.kill()
    _, alive = psutil.wait_procs(tree[1:], timeout=1.0)
    for c in alive:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            pass
    return True


def kill_all(rows: list[dict]):
    import psutil
    procs = []
    for r in rows:
        try:
            procs.append(psutil.Process(r["pid"]))
        except psutil.NoSuchProcess:
            pass
    for p in procs:
        p.terminate()
    _, alive = psutil.wait_procs(procs, timeout=2.0)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
