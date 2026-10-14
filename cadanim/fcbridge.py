"""Process boundary between the pipeline (system Python) and FreeCAD.

FreeCAD ships its own Python, so geometry work runs in ``freecadcmd``
subprocesses executing ``cadanim/fc/worker.py``. This module is the only place
that knows how to launch them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FREECADCMD = "/opt/freecad/squashfs-root/usr/bin/freecadcmd"


class FreeCADError(RuntimeError):
    pass


def freecadcmd_path() -> str:
    for cand in (os.environ.get("CADANIM_FREECADCMD"), shutil.which("freecadcmd"),
                 shutil.which("FreeCADCmd"), DEFAULT_FREECADCMD):
        if cand and os.path.exists(cand):
            return cand
    raise FreeCADError(
        "freecadcmd not found. Run scripts/install_freecad.sh or set CADANIM_FREECADCMD.")


def available() -> bool:
    try:
        freecadcmd_path()
        return True
    except FreeCADError:
        return False


_BOOTSTRAP = (
    "import sys, runpy; sys.path.insert(0, {root!r}); "
    "runpy.run_module('cadanim.fc.worker', run_name='__main__')"
)


def run_job(job: Dict[str, Any], timeout: float = 3600) -> Dict[str, Any]:
    """Run one job in a fresh freecadcmd process and return its result dict."""
    job = dict(job)
    for key in ("source", "outdir"):
        if key in job:
            job[key] = os.path.abspath(job[key])
    with tempfile.TemporaryDirectory(prefix="cadanim_") as tmp:
        job_path = os.path.join(tmp, "job.json")
        res_path = os.path.join(tmp, "result.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh)
        env = dict(os.environ, CADANIM_JOB=job_path, CADANIM_RESULT=res_path)
        proc = subprocess.run(
            [freecadcmd_path(), "-c", _BOOTSTRAP.format(root=str(REPO_ROOT))],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, cwd=tmp,
        )
        if not os.path.exists(res_path):
            log = proc.stdout.decode("utf-8", "replace")[-4000:]
            raise FreeCADError(f"FreeCAD job '{job.get('action')}' produced no result "
                               f"(exit {proc.returncode}). Log tail:\n{log}")
        with open(res_path) as fh:
            result = json.load(fh)
    if not result.get("ok"):
        raise FreeCADError(f"FreeCAD job '{job.get('action')}' failed: {result.get('error')}\n"
                           f"{result.get('trace', '')}")
    return result


def run_jobs_parallel(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(run_job, jobs))
