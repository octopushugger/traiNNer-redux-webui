#!/usr/bin/env python3
"""traiNNer-redux-webui,FastAPI server"""

import argparse
import asyncio
import math
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

BASE_DIR        = Path(__file__).parent          # webui/
ROOT_DIR        = BASE_DIR.parent                # project root (contains traiNNer-redux/ by default)
# Allow overriding the traiNNer-redux location via env var (set by launch scripts).
TRAINNER_DIR    = Path(os.environ.get("TRAINNER_REDUX_DIR")
                       or (ROOT_DIR / "traiNNer-redux")).resolve()
INSTANCES_FILE  = BASE_DIR / "instances.json"
TEMPLATES_DIR   = TRAINNER_DIR / "options" / "_templates" / "train"
TRAIN_DIR       = TRAINNER_DIR / "options" / "train"
TB_DIR          = TRAINNER_DIR / "tb_logger"
CLOUDFLARED_DIR = BASE_DIR / "cloudflared_bin"

app = FastAPI(title="traiNNer-redux-webui")

# ── In-memory state ─────────────────────────────────────────────────────────
training_procs:     dict[str, asyncio.subprocess.Process] = {}
stop_files:         dict[str, Path]                       = {}   # key → sentinel file path
ws_connections:     dict[str, list[WebSocket]]            = {}
manually_stopped:   set[str]                              = set()  # keys stopped by user

# TensorBoard graph cache: key → (mtime, scalars_dict)
_tb_cache: dict[str, tuple[float, dict]] = {}

# ── Persistence helpers ──────────────────────────────────────────────────────
# instances.json is keyed by the resolved TRAINNER_DIR path so that switching
# to a different traiNNer-redux directory never mixes up experiments.
# File format:
#   { "dirs": { "/abs/path/trainner-redux": { "Arch/exp": {...}, ... }, ... } }
# Legacy files with the old flat { "instances": {...} } format are migrated
# automatically by folding the existing entries under the current TRAINNER_DIR.

_TRAINNER_DIR_KEY = str(TRAINNER_DIR)

def load_experiments() -> dict:
    """Return {"instances": {…}} for the active TRAINNER_DIR only."""
    if not INSTANCES_FILE.exists():
        return {"instances": {}}
    raw = json.loads(INSTANCES_FILE.read_text(encoding="utf-8"))
    if "dirs" in raw:
        return {"instances": raw["dirs"].get(_TRAINNER_DIR_KEY, {})}
    # ── Legacy migration: old flat format ────────────────────────────────────
    # Wrap the existing entries under the current TRAINNER_DIR and rewrite.
    migrated = {"dirs": {_TRAINNER_DIR_KEY: raw.get("instances", {})}}
    INSTANCES_FILE.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
    return {"instances": migrated["dirs"][_TRAINNER_DIR_KEY]}

def save_experiments(data: dict) -> None:
    """Persist {"instances": {…}} back into the multi-dir file."""
    raw = json.loads(INSTANCES_FILE.read_text(encoding="utf-8")) \
          if INSTANCES_FILE.exists() else {}
    if "dirs" not in raw:
        raw = {"dirs": {}}
    raw["dirs"][_TRAINNER_DIR_KEY] = data["instances"]
    INSTANCES_FILE.write_text(json.dumps(raw, indent=2), encoding="utf-8")

def _read_config_name(yml: Path) -> "str | None":
    """Read the `name:` field from a YAML config file."""
    try:
        content = yml.read_text(encoding="utf-8")
        m = re.search(r"^name\s*:\s*(\S+)", content, re.MULTILINE)
        return m.group(1).strip() if m else None
    except OSError:
        return None

def _detect_arch(yml: Path) -> str:
    """Best-effort arch detection from network_g.type in a YAML config."""
    try:
        content = yml.read_text(encoding="utf-8")
        m = re.search(r"network_g\s*:[^\n]*\n\s+type\s*:\s*(\w+)", content)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "imported"

def _file_created_at(path: Path) -> str:
    """Return the older of ctime/mtime as an ISO string.

    File copies on some systems reset ctime to the copy date, so taking the
    minimum gives the true original date when mtime was preserved.
    """
    try:
        st = path.stat()
        return datetime.fromtimestamp(min(st.st_ctime, st.st_mtime)).isoformat()
    except OSError:
        return datetime.now().isoformat()

def _tb_last_modified(tb_dir: Path) -> "str | None":
    """Return the mtime of the most recently modified file under a tb_logger dir."""
    if not tb_dir.is_dir():
        return None
    latest: "float | None" = None
    for f in tb_dir.rglob("*"):
        if f.is_file():
            try:
                mtime = f.stat().st_mtime
                if latest is None or mtime > latest:
                    latest = mtime
            except OSError:
                pass
    return datetime.fromtimestamp(latest).isoformat() if latest is not None else None

def _scan_and_register(data: dict) -> bool:
    """Auto-discover config files not yet tracked in instances.json.

    Scans two locations (in priority order):
      1. options/train/<arch>/<name>.yml  ,the canonical source
      2. experiments/<name>/<name>.yml    ,fallback for configs trained outside the GUI

    Returns True if any new experiments were added (so the caller can persist).
    """
    changed = False
    # Build a set of already-known config paths for quick lookup
    known_paths = {Path(inst["config_path"]).resolve()
                   for inst in data["instances"].values()}

    # ── 1. options/train/<arch>/<name>.yml ────────────────────────────────────
    if TRAIN_DIR.exists():
        for yml in TRAIN_DIR.rglob("*.yml"):
            if yml.resolve() in known_paths:
                continue
            try:
                rel   = yml.relative_to(TRAIN_DIR)
                parts = rel.parts
            except ValueError:
                continue
            if len(parts) != 2:
                continue   # skip unexpected nesting
            arch, _fn = parts
            name = yml.stem
            key  = f"{arch}/{name}"
            if key in data["instances"] and not data["instances"][key].get("deleted"):
                continue
            cfg_name = _read_config_name(yml) or name
            tb_dir   = TB_DIR / cfg_name
            data["instances"][key] = {
                "key":               key,
                "name":              name,
                "arch":              arch,
                "config_path":       str(yml),
                "created_at":        _file_created_at(yml),
                "last_run_at":       _tb_last_modified(tb_dir),
                "deleted":           False,
                "tb_log_dir":        str(tb_dir),
                "timestamps_verified": True,
            }
            known_paths.add(yml.resolve())
            changed = True

    # ── 2. experiments/<name>/*.yml (fallback) ────────────────────────────────
    # Only import configs for which NO matching options/train/<arch>/<cfg_name>.yml
    # exists anywhere,i.e. experiments trained entirely outside the GUI that
    # have no canonical options/ entry.
    exp_base = TRAINNER_DIR / "experiments"
    if exp_base.exists():
        for exp_dir in sorted(exp_base.iterdir()):
            if not exp_dir.is_dir():
                continue
            for yml in sorted(exp_dir.glob("*.yml")):
                if yml.resolve() in known_paths:
                    continue
                cfg_name = _read_config_name(yml) or yml.stem

                # Skip if options/train/ already has a config whose YAML name: field
                # matches,either by filename (fast path) or by reading each file
                # (handles cases where filename ≠ name: field).
                # Pass 1 will have already registered whichever file that is.
                options_covered = False
                if TRAIN_DIR.exists():
                    # Fast path: filename stem matches
                    if any(TRAIN_DIR.glob(f"*/{cfg_name}.yml")):
                        options_covered = True
                    else:
                        # Slow path: scan registered instances whose config_path is
                        # inside options/train/ and whose YAML name: matches
                        for inst in data["instances"].values():
                            if inst.get("deleted"):
                                continue
                            cp = Path(inst["config_path"])
                            try:
                                cp.relative_to(TRAIN_DIR)   # raises if not under TRAIN_DIR
                            except ValueError:
                                continue
                            if _read_config_name(cp) == cfg_name:
                                options_covered = True
                                break
                if options_covered:
                    continue

                # Skip if already registered under this name for any other reason
                if any(not inst.get("deleted") and inst.get("name") == cfg_name
                       for inst in data["instances"].values()):
                    continue

                arch = _detect_arch(yml)
                name = cfg_name

                # Prefer an existing options/train/<arch>/ directory whose name
                # matches case-insensitively; only fall back to creating a new
                # directory (with whatever case _detect_arch returned) when none
                # exists at all.
                if TRAIN_DIR.exists():
                    arch_lower = arch.lower()
                    for existing in TRAIN_DIR.iterdir():
                        if existing.is_dir() and existing.name.lower() == arch_lower:
                            arch = existing.name
                            break

                key  = f"{arch}/{name}"
                # Ensure unique key in the unlikely event of a collision
                base_key, counter = key, 1
                while key in data["instances"] and not data["instances"][key].get("deleted"):
                    key = f"{base_key}_{counter}"; counter += 1

                # Copy config into options/train/<arch>/<name>.yml so it becomes
                # a proper first-class entry and future scans pick it up via pass 1.
                # Never overwrite an existing file,someone may have a different
                # config there (e.g. they changed name: without renaming the file).
                dest_dir  = TRAIN_DIR / arch
                dest_path = dest_dir / f"{name}.yml"
                if dest_path.exists():
                    config_path = yml
                else:
                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(yml, dest_path)
                        config_path = dest_path
                        known_paths.add(dest_path.resolve())
                    except OSError:
                        config_path = yml

                tb_dir2 = TB_DIR / name
                data["instances"][key] = {
                    "key":               key,
                    "name":              name,
                    "arch":              arch,
                    "config_path":       str(config_path),
                    "created_at":        _file_created_at(yml),
                    "last_run_at":       _tb_last_modified(tb_dir2),
                    "deleted":           False,
                    "tb_log_dir":        str(tb_dir2),
                    "timestamps_verified": True,
                }
                known_paths.add(config_path.resolve())
                changed = True

    return changed

# ── Request/response models ──────────────────────────────────────────────────

class NewExperimentRequest(BaseModel):
    arch: str
    name: str

class RenameRequest(BaseModel):
    new_name: str

class ConfigSaveRequest(BaseModel):
    content: str

class StatsRequest(BaseModel):
    epoch:  Optional[int]   = None
    iter:   Optional[int]   = None
    its:    Optional[float] = None
    eta:    Optional[str]   = None
    lr_val: Optional[float] = None
    vram:   Optional[float] = None

def _fix_timestamps(data: dict) -> bool:
    """One-time back-fill of created_at / last_run_at for legacy entries.

    created_at: only touched on entries that lack the timestamps_verified flag
    (i.e. entries that were registered before filesystem-based dating was added).
    Once fixed the flag is set so the value is never overwritten again, even if
    the directory is copied and file timestamps change.

    last_run_at: filled in from the tb_logger directory whenever it is None,
    regardless of the flag, since training may have run before the GUI tracked it.
    """
    changed = False
    for inst in data["instances"].values():
        if inst.get("deleted"):
            continue
        if not inst.get("timestamps_verified"):
            config_path = Path(inst["config_path"])
            if config_path.exists():
                fs_date = _file_created_at(config_path)
                if not inst.get("created_at") or fs_date < inst["created_at"]:
                    inst["created_at"] = fs_date
                    changed = True
            inst["timestamps_verified"] = True
            changed = True
        if inst.get("last_run_at") is None:
            tb_dir = _find_tb_log_dir(inst)
            if tb_dir:
                last = _tb_last_modified(tb_dir)
                if last:
                    inst["last_run_at"] = last
                    changed = True
    return changed

# ── API: experiments ──────────────────────────────────────────────────────────

@app.get("/api/experiments")
def get_experiments():
    data = load_experiments()
    dirty  = _scan_and_register(data)
    dirty |= _fix_timestamps(data)
    if dirty:
        save_experiments(data)
    experiments = [v for v in data["instances"].values() if not v.get("deleted")]
    return {"experiments": experiments}

@app.get("/api/archs")
def get_archs():
    if not TEMPLATES_DIR.exists():
        return {"archs": []}
    archs = sorted(p.name for p in TEMPLATES_DIR.iterdir() if p.is_dir())
    return {"archs": archs}

@app.get("/api/archs/{arch}/templates")
def get_arch_templates(arch: str):
    arch_dir = TEMPLATES_DIR / arch
    if not arch_dir.exists():
        return {"templates": []}
    templates = sorted(p.stem for p in arch_dir.glob("*.yml"))
    return {"templates": templates}

@app.post("/api/experiments")
def create_experiment(req: NewExperimentRequest):
    data = load_experiments()
    key = f"{req.arch}/{req.name}"
    if key in data["instances"] and not data["instances"][key].get("deleted"):
        raise HTTPException(400, "Experiment already exists")

    config_dir = TRAIN_DIR / req.arch
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{req.name}.yml"

    # Copy first matching template, or create a stub
    tmpl_dir = TEMPLATES_DIR / req.arch
    tmpl_path: Optional[Path] = None
    if tmpl_dir.exists():
        tmpls = sorted(tmpl_dir.glob("*.yml"))
        if tmpls:
            tmpl_path = tmpls[0]

    if tmpl_path and tmpl_path.exists():
        content = tmpl_path.read_text(encoding="utf-8")
        content = re.sub(r'^(name\s*:\s*).*$', rf'\g<1>{req.name}', content,
                         count=1, flags=re.MULTILINE)
        config_path.write_text(content, encoding="utf-8")
    else:
        config_path.write_text(f"# {req.arch} training config\nname: {req.name}\n", encoding="utf-8")

    experiment = {
        "key": key,
        "name": req.name,
        "arch": req.arch,
        "config_path": str(config_path),
        "created_at": datetime.now().isoformat(),
        "last_run_at": None,
        "deleted": False,
        "tb_log_dir": str(TB_DIR / req.name),
    }
    data["instances"][key] = experiment
    save_experiments(data)
    return experiment

@app.delete("/api/experiments/{key:path}")
def delete_experiment(key: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    data["instances"][key]["deleted"] = True
    save_experiments(data)
    return {"ok": True}

@app.post("/api/experiments/{key:path}/duplicate")
def duplicate_experiment(key: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    src = data["instances"][key]

    new_name = src["name"] + "_copy"
    new_key  = f"{src['arch']}/{new_name}"
    counter  = 1
    while new_key in data["instances"] and not data["instances"][new_key].get("deleted"):
        new_name = f"{src['name']}_copy{counter}"
        new_key  = f"{src['arch']}/{new_name}"
        counter += 1

    src_config = Path(src["config_path"])
    new_config  = src_config.parent / f"{new_name}.yml"
    if src_config.exists():
        shutil.copy(src_config, new_config)
    else:
        new_config.write_text("", encoding="utf-8")

    experiment = {
        **src,
        "key": new_key,
        "name": new_name,
        "config_path": str(new_config),
        "created_at": datetime.now().isoformat(),
        "last_run_at": None,
        "deleted": False,
        "tb_log_dir": str(TB_DIR / new_name),
    }
    data["instances"][new_key] = experiment
    save_experiments(data)
    return experiment

@app.post("/api/experiments/{key:path}/rename")
def rename_experiment(key: str, req: RenameRequest):
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")
    if "/" in new_name or "\\" in new_name:
        raise HTTPException(400, "Name cannot contain path separators")

    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")

    inst     = data["instances"][key]
    old_name = inst["name"]
    arch     = inst["arch"]
    new_key  = f"{arch}/{new_name}"

    if new_name == old_name:
        return inst  # no-op

    if new_key in data["instances"] and not data["instances"][new_key].get("deleted"):
        raise HTTPException(400, "An experiment with that name already exists")

    old_config_path = Path(inst["config_path"])

    # Read the config's name: field BEFORE any file operations so it's available
    # for locating training-created directories (experiments/ and tb_logger/ use
    # the YAML name: field, not the GUI experiment name).
    old_cfg_name = _read_config_name(old_config_path)

    # 1. Update YAML name field and move config file
    new_config_path = old_config_path.parent / f"{new_name}.yml"
    if old_config_path.exists():
        content = old_config_path.read_text(encoding="utf-8")
        content = re.sub(
            r'^(name\s*:\s*).*$', rf'\g<1>{new_name}', content,
            count=1, flags=re.MULTILINE,
        )
        new_config_path.write_text(content, encoding="utf-8")
        if old_config_path != new_config_path:
            old_config_path.unlink()
    # else: config file doesn't exist yet; new_config_path will be empty stub

    # 2. Rename experiments output folder if it exists.
    # traiNNer-redux names this folder after the config's name: field, which may
    # differ from the GUI experiment name, so try both.
    exp_base    = TRAINNER_DIR / "experiments"
    new_exp_dir = exp_base / new_name
    old_exp_dir = None
    for candidate in dict.fromkeys(n for n in [old_name, old_cfg_name] if n):
        p = exp_base / candidate
        if p.exists():
            old_exp_dir = p
            break
    if old_exp_dir and old_exp_dir != new_exp_dir:
        old_exp_dir.rename(new_exp_dir)
        # Update name: in any YML files saved inside the experiments dir so that
        # _scan_and_register doesn't re-register them as a new experiment on the
        # next page load.
        for yml in new_exp_dir.rglob("*.yml"):
            try:
                text    = yml.read_text(encoding="utf-8")
                updated = re.sub(
                    r'^(name\s*:\s*).*$', rf'\g<1>{new_name}', text,
                    count=1, flags=re.MULTILINE,
                )
                if updated != text:
                    yml.write_text(updated, encoding="utf-8")
            except OSError:
                pass

    # 3. Rename tb_logger folder if it exists.
    # Use old_cfg_name (captured before step 1 deleted the file) to find the
    # real log directory, then fall back to the GUI name.
    new_tb_dir    = TB_DIR / new_name
    actual_tb_dir = None
    for candidate in dict.fromkeys(n for n in [old_cfg_name, old_name] if n):
        p = TB_DIR / candidate
        if p.is_dir():
            actual_tb_dir = p
            break
    if actual_tb_dir and actual_tb_dir != new_tb_dir:
        actual_tb_dir.rename(new_tb_dir)

    # 4. Update instances.json
    del data["instances"][key]
    inst["key"]         = new_key
    inst["name"]        = new_name
    inst["config_path"] = str(new_config_path)
    inst["tb_log_dir"]  = str(new_tb_dir)
    data["instances"][new_key] = inst
    save_experiments(data)

    return inst

@app.put("/api/experiments/{key:path}/stats")
def save_stats(key: str, req: StatsRequest):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    data["instances"][key]["stats"] = req.model_dump()
    save_experiments(data)
    return {"ok": True}

@app.get("/api/experiments/{key:path}/config")
def get_config(key: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    config_path = Path(data["instances"][key]["config_path"])
    if not config_path.exists():
        return {"content": ""}
    return {"content": config_path.read_text(encoding="utf-8")}

@app.put("/api/experiments/{key:path}/config")
def save_config(key: str, req: ConfigSaveRequest):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    config_path = Path(data["instances"][key]["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(req.content, encoding="utf-8")
    return {"ok": True}

# ── API: training ─────────────────────────────────────────────────────────────

@app.get("/api/training")
def get_training():
    """Return the currently running experiment key, if any."""
    key = next(iter(training_procs), None)
    return {"key": key}

@app.get("/api/experiments/{key:path}/status")
def get_status(key: str):
    return {"running": key in training_procs}

@app.get("/api/experiments/{key:path}/graphs")
def get_graphs(key: str):
    """Return all scalar metrics from the TensorBoard log for this experiment."""
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")

    inst = data["instances"][key]

    if not _HAS_TB:
        return {"scalars": {}, "error": "tensorboard package not installed"}

    tb_dir = _find_tb_log_dir(inst)

    if tb_dir is None:
        cfg_name = _config_name(inst)
        names_tried = []
        if cfg_name:
            names_tried.append(cfg_name)
        if inst["name"] not in names_tried:
            names_tried.append(inst["name"])
        tb_subdirs = sorted(p.name for p in TB_DIR.iterdir() if p.is_dir()) if TB_DIR.is_dir() else []
        return {
            "scalars": {},
            "tried":   [str(TB_DIR / n) for n in names_tried],
            "tb_subdirs": tb_subdirs,
        }

    event_files = list(tb_dir.rglob("events.out.tfevents*"))
    if not event_files:
        return {"scalars": {}, "error": "no event files in directory"}

    mtime = max(f.stat().st_mtime for f in event_files)

    cached = _tb_cache.get(key)
    if cached and cached[0] >= mtime:
        return {"scalars": cached[1]}

    # Read events (blocking I/O,acceptable for a local tool)
    try:
        ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
        ea.Reload()
        all_tags = ea.Tags()
        scalars: dict[str, list] = {}
        for tag in sorted(all_tags.get("scalars", [])):
            events = ea.Scalars(tag)
            pts = [{"step": e.step, "value": float(e.value)}
                   for e in events if math.isfinite(e.value)]
            # Uniform downsample to ≤2000 points so JSON stays small
            if len(pts) > 2000:
                s = len(pts) / 2000
                pts = [pts[int(i * s)] for i in range(2000)]
            scalars[tag] = pts
    except Exception as exc:
        print(f"[graphs] EventAccumulator error: {exc}")
        return {"scalars": {}, "tb_dir": str(tb_dir), "error": str(exc)}

    _tb_cache[key] = (mtime, scalars)
    return {"scalars": scalars}

def _config_name(inst: dict) -> str | None:
    """Read the 'name:' field from the experiment's YAML config."""
    return _read_config_name(Path(inst["config_path"]))

def _find_tb_log_dir(inst: dict) -> "Path | None":
    """Locate the TensorBoard log directory under tb_logger/.

    traiNNer-redux always writes logs to tb_logger/<name>/ where <name> is
    the 'name:' field in the YAML config at the time training ran.
    We try the config's name: field first, then the GUI experiment name.
    """
    if not TB_DIR.is_dir():
        return None

    # Priority: config name: field → GUI experiment name
    names = []
    cfg_name = _config_name(inst)
    if cfg_name:
        names.append(cfg_name)
    if inst["name"] not in names:
        names.append(inst["name"])

    for name in names:
        p = TB_DIR / name
        if p.is_dir():
            return p

    return None

# ── API: visualization images ─────────────────────────────────────────────────

_VIZ_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}

def _viz_dir(inst: dict) -> Path:
    """Return the visualization directory for an experiment.

    traiNNer-redux names the experiments/ output folder after the config's
    name: field, which may differ from the GUI experiment name.  Try the
    config name first, then fall back to the GUI experiment name.
    """
    exp_base = TRAINNER_DIR / "experiments"
    names = []
    cfg_name = _config_name(inst)
    if cfg_name:
        names.append(cfg_name)
    if inst["name"] not in names:
        names.append(inst["name"])
    for name in names:
        p = exp_base / name / "visualization"
        if p.is_dir():
            return p
    # Default to config name (or GUI name) even if the dir doesn't exist yet.
    return exp_base / names[0] / "visualization"

def _parse_val_gt_dirs(config_path: Path) -> list[Path]:
    """Return all resolved dataroot_gt paths found in a config file."""
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError:
        return []
    result = []
    # Match dataroot_gt: value — value may be a scalar, inline list [...], or multi-line list
    for m in re.finditer(r'dataroot_gt\s*:\s*(.*)', content):
        rest = m.group(1).strip()
        if rest.startswith('['):
            # Check if closed on same line
            if ']' in rest:
                inner = rest[1:rest.index(']')]
            else:
                # Multi-line list: collect until closing ]
                after = content[m.end():]
                close = after.find(']')
                inner = (rest[1:] + (',' if rest[1:].strip() else '') + after[:close]) if close != -1 else rest[1:]
            raw_items = re.split(r'[\n,]+', inner)
        else:
            raw_items = [rest]
        for item in raw_items:
            p = item.strip().strip('"\'').split('#')[0].strip()
            if not p:
                continue
            p = os.path.expanduser(p)
            resolved = Path(p) if Path(p).is_absolute() else (TRAINNER_DIR / p).resolve()
            result.append(resolved)
    return result

@app.get("/api/experiments/{key:path}/visualization")
def get_visualization(key: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    inst    = data["instances"][key]
    viz_dir = _viz_dir(inst)
    if not viz_dir.is_dir():
        return {"folders": []}

    def _sort_key(p: Path):
        return int(p.name) if p.name.isdigit() else p.name

    _LR_SUFFIXES = ('_lq', '_lr')

    folders = []
    for folder in sorted(viz_dir.iterdir(), key=_sort_key):
        if not folder.is_dir():
            continue

        all_files = {
            f.stem: f.name
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in _VIZ_EXTS
        }

        # Any file whose stem ends with a known LR suffix is an LR file.
        def _is_lr(stem: str) -> bool:
            return any(stem.endswith(s) for s in _LR_SUFFIXES)

        sr_images = sorted(fname for stem, fname in all_files.items() if not _is_lr(stem))
        lq_file_names = {fname for stem, fname in all_files.items() if _is_lr(stem)}

        # Build per-image LR map: SR image → its LR counterpart.
        # Try each known suffix for an exact stem match first.
        lq_map: dict[str, str] = {}
        for sr_stem, sr_name in all_files.items():
            if _is_lr(sr_stem):
                continue
            for suf in _LR_SUFFIXES:
                lq_stem = sr_stem + suf
                if lq_stem in all_files:
                    lq_map[sr_name] = all_files[lq_stem]
                    break
            else:
                # No per-image match — if exactly one LR file exists in the
                # folder use it as the LR for every SR image (common when
                # traiNNer-redux saves a single representative LQ input).
                if len(lq_file_names) == 1:
                    lq_map[sr_name] = next(iter(lq_file_names))

        if sr_images:
            folders.append({"name": folder.name, "images": sr_images, "lq_map": lq_map})
    return {"folders": folders}

@app.get("/api/experiments/{key:path}/visualization/{folder}/{filename}")
def get_viz_image(key: str, folder: str, filename: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    inst    = data["instances"][key]
    vdir    = _viz_dir(inst).resolve()
    target  = (vdir / folder / filename).resolve()
    try:
        target.relative_to(vdir)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(target))

@app.get("/api/experiments/{key:path}/gt-debug")
def get_gt_debug(key: str, filename: str = ""):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    inst = data["instances"][key]
    config_path = Path(inst["config_path"])
    gt_dirs = _parse_val_gt_dirs(config_path)
    result: dict = {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "gt_dirs": [],
    }
    for d in gt_dirs:
        entry: dict = {"path": str(d), "exists": d.is_dir(), "sample_files": []}
        if d.is_dir():
            entry["sample_files"] = sorted(f.name for f in d.iterdir())[:20]
        if filename and d.is_dir():
            stem = Path(filename).stem
            orig_ext = Path(filename).suffix
            for ext in ([orig_ext] if orig_ext else []) + ['.png', '.jpg', '.jpeg', '.webp']:
                candidate = d / (stem + ext)
                entry.setdefault("candidates_tried", []).append(
                    {"path": str(candidate), "exists": candidate.exists()}
                )
        result["gt_dirs"].append(entry)
    return result

@app.get("/api/experiments/{key:path}/gt-image/{filename}")
def get_gt_image(key: str, filename: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    inst      = data["instances"][key]
    stem      = Path(filename).stem
    orig_ext  = Path(filename).suffix
    # traiNNer names SR outputs as {img_name}_{iteration}.png — strip the iteration suffix
    base_stem = re.sub(r'_\d{4,}$', '', stem)
    stems_to_try = list(dict.fromkeys([stem, base_stem]))  # dedup, original first
    gt_dirs   = _parse_val_gt_dirs(Path(inst["config_path"]))
    for gt_dir in gt_dirs:
        if not gt_dir.is_dir():
            continue
        gt_dir_r = gt_dir.resolve()
        for s in stems_to_try:
            for ext in ([orig_ext] if orig_ext else []) + ['.png', '.jpg', '.jpeg', '.webp']:
                candidate = (gt_dir / (s + ext)).resolve()
                try:
                    candidate.relative_to(gt_dir_r)
                except ValueError:
                    continue
                if candidate.exists():
                    return FileResponse(str(candidate))
    raise HTTPException(404, "GT image not found")

@app.post("/api/experiments/{key:path}/start")
async def start_training(key: str):
    data = load_experiments()
    if key not in data["instances"]:
        raise HTTPException(404, "Experiment not found")
    if key in training_procs:
        raise HTTPException(400, "Already running")

    inst        = data["instances"][key]
    config_path = inst["config_path"]
    train_cwd   = str(TRAINNER_DIR)
    python      = sys.executable

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # live output without block-buffering
    env["FORCE_COLOR"]      = "1"   # rich honours this even when stdout is a pipe
    env["COLORTERM"]        = "truecolor"

    # Sentinel file: the stop endpoint touches this file; rich_ansi_patch.py
    # watches for it and raises KeyboardInterrupt in the training main thread.
    stop_file = BASE_DIR / f".stop_{hashlib.md5(key.encode()).hexdigest()[:12]}"
    stop_file.unlink(missing_ok=True)   # clean up any leftover from a previous run
    env["TRAIINNER_STOP_FILE"] = str(stop_file)

    wrapper = str(BASE_DIR / "rich_ansi_patch.py")
    proc = await asyncio.create_subprocess_exec(
        python, "-u", wrapper, "train.py", "--auto_resume", "-opt", config_path,
        cwd=train_cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    training_procs[key] = proc
    stop_files[key]     = stop_file

    data["instances"][key]["last_run_at"] = datetime.now().isoformat()
    save_experiments(data)

    asyncio.create_task(_stream_output(key, proc))
    return {"ok": True, "pid": proc.pid}

@app.post("/api/experiments/{key:path}/stop")
async def stop_training(key: str):
    if key not in training_procs:
        raise HTTPException(400, "Not running")
    manually_stopped.add(key)
    # Touch the sentinel file,the watcher thread in rich_ansi_patch.py detects
    # it and raises KeyboardInterrupt in the training main thread, triggering
    # traiNNer-redux's save-on-exit handler without any signal/native-runtime issues.
    if key in stop_files:
        stop_files[key].touch()
    return {"ok": True}

_srv_line:      str = ""   # content being built for the current in-progress line
_srv_displayed: str = ""   # what was last written to the terminal for that line
# Strip all non-SGR CSI sequences (cursor movement, erase, etc.) but keep
# SGR colour codes (ending in 'm').
_NON_SGR  = re.compile(r"\x1b\[[0-9;]*[A-LN-Za-ln-z]|\x1b[^[]")
# Matches a tqdm progress-bar line: "N% -----"
_TQDM_BAR = re.compile(r"\d+%\s+[-]{5,}")

def _mirror_to_console(text: str) -> None:
    """Write training output to the server console.

    tqdm bar lines (matching N% ---) are rendered in-place with CR so repeated
    updates overwrite each other on one terminal line.  All other lines (rich
    log messages, metrics, etc.) are committed with a newline as normal.
    """
    global _srv_line, _srv_displayed
    text = _NON_SGR.sub("", text)
    parts = re.split(r"(\r\n|\r|\n)", text)
    for p in parts:
        if p in ("\r\n", "\n"):
            line = _srv_line
            _srv_line = ""
            if not line:
                continue
            if _TQDM_BAR.search(line):
                if line != _srv_displayed:
                    pad = max(0, len(_srv_displayed) - len(line))
                    sys.stdout.write("\r" + line + (" " * pad))
                    sys.stdout.flush()
                    _srv_displayed = line
            else:
                pad = max(0, len(_srv_displayed) - len(line))
                sys.stdout.write("\r" + line + (" " * pad) + "\n")
                sys.stdout.flush()
                _srv_displayed = ""
        elif p == "\r":
            _srv_line = ""
        elif p:
            _srv_line += p
    if _srv_line and _srv_line != _srv_displayed and _TQDM_BAR.search(_srv_line):
        pad = max(0, len(_srv_displayed) - len(_srv_line))
        sys.stdout.write("\r" + _srv_line + (" " * pad))
        sys.stdout.flush()
        _srv_displayed = _srv_line

async def _stream_output(key: str, proc: asyncio.subprocess.Process):
    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            _mirror_to_console(text)
            await _broadcast(key, text)
    except Exception:
        pass
    finally:
        training_procs.pop(key, None)
        sf = stop_files.pop(key, None)
        if sf:
            sf.unlink(missing_ok=True)
        was_manual  = key in manually_stopped
        manually_stopped.discard(key)
        exit_code   = proc.returncode if proc.returncode is not None else -1

        if was_manual:
            sentinel = "manual_stop"
        elif exit_code == 0:
            sentinel = "complete"
        else:
            sentinel = "stopped"

        await _broadcast(key, "\n[Process ended]\n")
        await _broadcast(key, f"\x00STATUS:{sentinel}\n")

async def _broadcast(key: str, text: str):
    dead = []
    for ws in ws_connections.get(key, []):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[key].remove(ws)

# ── WebSocket: console stream ─────────────────────────────────────────────────

@app.websocket("/ws/{key:path}")
async def websocket_endpoint(websocket: WebSocket, key: str):
    await websocket.accept()
    ws_connections.setdefault(key, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        pass
    finally:
        conns = ws_connections.get(key, [])
        if websocket in conns:
            conns.remove(websocket)

# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

# ── Cloudflare Quick Tunnel ───────────────────────────────────────────────────

def _cloudflared_path() -> Path:
    CLOUDFLARED_DIR.mkdir(exist_ok=True)
    system = platform.system().lower()
    if system == "windows":
        return CLOUDFLARED_DIR / "cloudflared.exe"
    return CLOUDFLARED_DIR / "cloudflared"

def _download_cloudflared() -> Path:
    dest   = _cloudflared_path()
    system = platform.system().lower()
    machine = platform.machine().lower()

    if dest.exists():
        return dest

    if system == "windows":
        filename = "cloudflared-windows-amd64.exe"
    elif system == "linux":
        filename = "cloudflared-linux-arm64" if ("arm" in machine or "aarch64" in machine) else "cloudflared-linux-amd64"
    elif system == "darwin":
        filename = "cloudflared-darwin-amd64.tgz"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{filename}"
    print(f"Downloading cloudflared: {url}")
    urllib.request.urlretrieve(url, dest)

    if system != "windows":
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    return dest

async def _start_cloudflare_tunnel(port: int):
    import re
    try:
        cf = _download_cloudflared()
    except Exception as e:
        print(f"[cloudflared] Download failed: {e}")
        return

    proc = await asyncio.create_subprocess_exec(
        str(cf), "tunnel", "--url", f"http://localhost:{port}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    url_pattern = re.compile(r"https://[\w-]+\.trycloudflare\.com")
    print("Waiting for Cloudflare tunnel URL…")
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        m = url_pattern.search(line)
        if m:
            bar = "=" * 60
            print(f"\n{bar}\n  Public URL: {m.group()}\n{bar}\n")
            break

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="traiNNer-redux-webui")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--online", action="store_true",
                        help="Expose publicly via Cloudflare Quick Tunnel")
    args = parser.parse_args()

    if args.online:
        args.host = "0.0.0.0"

    local_url = f"http://localhost:{args.port}"
    print(f"\n traiNNer-redux-webui")
    print(f" Local: {local_url}\n")

    if args.online:
        async def _run():
            asyncio.create_task(_start_cloudflare_tunnel(args.port))
            cfg    = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
            server = uvicorn.Server(cfg)
            await server.serve()
        asyncio.run(_run())
    else:
        # Open browser after a brief delay.
        # On Linux webbrowser.open() ignores the desktop default and walks its
        # own hardcoded list (Firefox first); use xdg-open instead so the
        # system default browser is respected.
        import threading, time
        def _open():
            time.sleep(1.2)
            if platform.system() == "Linux":
                subprocess.Popen(["xdg-open", local_url],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import webbrowser
                webbrowser.open(local_url)
        threading.Thread(target=_open, daemon=True).start()
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
