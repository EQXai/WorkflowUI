#!/usr/bin/env python3
"""Command-line interface to run ComfyUI workflows using a saved GUI preset.

Usage examples
--------------
Run a preset exactly as saved:
    python orchestrator_cli.py MyPreset.json

Run the same preset but aim for 55 successful final images:
    python orchestrator_cli.py MyPreset.json 55

The script locates presets relative to the *save_presets* folder if the given
path does not exist. It then loads the stored list of component values (the
same list Gradio stores) and forwards them to *run_pipeline_gui* from
*orchestrator_gui.py*.  A lightweight console progress is printed while the
pipeline executes.  No Gradio UI is launched.
"""
from __future__ import annotations

import argparse
import json
import sys
import re
import os
from pathlib import Path
from typing import List, Any
import time

# Local modules – orchestrator_gui contains the heavy lifting.
import orchestrator_gui as gui  # noqa: E402

from datetime import datetime

# Attempt to import rich for fancy CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
except ImportError:
    Console = None  # Fallback if rich not installed

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PRESET_DIR = THIS_DIR / "save_presets"
BATCH_RUNS_INDEX = 9  # Index of *batch_runs_input* inside ALL_COMPONENTS list

# Number of fixed parameters before dynamic_values in run_pipeline_gui
_FIXED_COUNT = 19
# Number of LoRA override arguments expected (always 24)
_LORA_COUNT = 24

# -----------------------------------------------------------------------------
# Replicate override mappings defined in the GUI so run_pipeline_gui can apply
# them even sin interfaz.
# -----------------------------------------------------------------------------
OVERRIDE_WF1 = [
    ("168", "steps", int),
    ("169", "steps", int),
    ("170", "text", str),
    ("173", "ckpt_name", str),
    ("176", "width", int),
    ("176", "height", int),
    ("180", "text", str),
    ("182", "steps", int),
]

OVERRIDE_WF2 = [
    ("248", "text", str),
    ("224", "guidance", float),
    ("226", "grain_power", float),
    ("230", "size", int),
    ("239", "steps", int),
    ("239", "denoise", float),
    ("242", "prompt", str),
    ("243", "prompt", str),
    ("287", "guidance", float),
    ("289", "grain_power", float),
    ("293", "size", int),
    ("302", "steps", int),
    ("302", "denoise", float),
    ("305", "prompt", str),
    ("306", "prompt", str),
    ("311", "text", str),
]

# Expose to orchestrator_gui so that run_pipeline_gui sees them
gui.WF1_MAPPING = OVERRIDE_WF1
gui.WF2_MAPPING = OVERRIDE_WF2

_OVERRIDES_LEN = len(OVERRIDE_WF1) + len(OVERRIDE_WF2)


def _load_preset_values(preset_path: str | Path) -> List[Any]:
    """Return the stored component values list for *preset_path*.

    If *preset_path* is not an existing file it is searched inside the default
    *save_presets* directory (adding .json if missing).
    """
    p = Path(preset_path)
    if not p.exists():
        # Try resolving inside save_presets folder
        candidate = PRESET_DIR / (str(p) if p.suffix else f"{p}.json")
        if candidate.exists():
            p = candidate
        else:
            sys.exit(f"[ERROR] Preset file not found: {preset_path}")

    try:
        return json.loads(p.read_text())
    except Exception as exc:  # pragma: no cover
        sys.exit(f"[ERROR] Could not read preset file '{p}': {exc}")


def _normalize_values(values: List[Any]) -> List[Any]:
    """Sanitize *values* so they match the positional parameters expected by
    *run_pipeline_gui* when executed head-less.

    Layout expected:
        0..18   – the 19 fixed arguments (see run_pipeline_gui signature)
        19..19+cat_len-1 – one boolean per NSFW category (excl. NOT_DETECTED)
        *MIDDLE* – (variable) workflow override widgets **to discard**
        last 24 – LoRA paths & strengths (keep if present)
    """
    cat_keys = [k for k in gui.CATEGORY_DISPLAY_MAPPINGS.keys() if k != "NOT_DETECTED"]
    cat_len = len(cat_keys)
    prefix_len = _FIXED_COUNT + cat_len

    lora_present = len(values) >= _LORA_COUNT
    lora_tail = values[-_LORA_COUNT:] if lora_present else []

    # Determine override segment
    start_override = prefix_len
    end_override = start_override + _OVERRIDES_LEN
    overrides_seg = values[start_override:end_override]

    # Pad overrides if missing
    if len(overrides_seg) < _OVERRIDES_LEN:
        overrides_seg += [None] * (_OVERRIDES_LEN - len(overrides_seg))

    sanitized = values[:prefix_len] + overrides_seg + lora_tail
    return sanitized


def _stream_pipeline(values: List[Any], expected_runs: int | None):
    """Run *run_pipeline_gui* and display a Rich Live table with progress."""

    if Console is None:
        print("[WARN] 'rich' not installed. Falling back to basic logging.")
        return _stream_pipeline_basic(values, expected_runs)

    console = Console()

    gen = gui.run_pipeline_gui(*values)

    # Data storage per attempt
    attempts: dict[int, dict[str, str]] = {}
    attempt_starts: dict[int, float] = {}
    current_attempt = 0
    # Track JSON collection per attempt
    collecting_check: dict[int, list[str]] = {}

    # Progress bar setup
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )

    progress_task = None
    if expected_runs:
        progress_task = progress.add_task("Approved", total=expected_runs)

    def _build_table() -> Table:
        saved_width = max(12, int(console.size.width * 0.2))  # 20 % ancho máx.
        table = Table(title="Pipeline Runs", expand=True)
        table.add_column("Attempt", justify="right")
        table.add_column("Status")
        table.add_column("PromptSeed", justify="right")
        table.add_column("ImageSeed", justify="right")
        table.add_column("Checks")
        table.add_column("Saved As", max_width=saved_width, overflow="ellipsis")
        return table

    def _render_group():
        tbl = _build_table()
        for at in sorted(attempts):
            row = attempts[at]

            def _color_status(s: str) -> str:
                mapping = {
                    "WF1": "yellow",
                    "Check": "cyan",
                    "WF2": "bright_blue",
                    "OK": "green",
                    "Queued": "grey50",
                    "Failed": "red",
                }
                color = mapping.get(s, "white")
                return f"[{color}]{s}[/{color}]" if s else s

            tbl.add_row(
                str(at),
                _color_status(row.get("Status", "")),
                str(row.get("PromptSeed", "")),
                str(row.get("ImageSeed", "")),
                row.get("Checks", ""),
                row.get("SavedAs", ""),
            )

        # ---- average as table caption ----
        durations = [v.get("Duration") for v in attempts.values() if v.get("Status") == "OK" and v.get("Duration")]
        avg_txt = "-" if not durations else f"{sum(durations)/len(durations):.1f}s"
        tbl.caption = f"[bold]Avg Time Per Image:[/bold] {avg_txt}"

        if progress_task is not None:
            from rich.console import Group
            return Group(progress, tbl)
        return tbl

    def _ensure_attempt(idx: int):
        if idx not in attempts:
            attempts[idx] = {
                "Status": "Queued",
                "PromptSeed": "-",
                "ImageSeed": "-",
                "Checks": "Pending",
                "SavedAs": "",
                "Duration": None,
            }

    renderable = _render_group()
    with Live(renderable, console=console, refresh_per_second=4, screen=False) as live:
        try:
            last_len = 0
            last_approved = 0

            for step in gen:
                log_text, pending, status, metrics = step[1], step[4], step[5], step[6]

                # Detect approved increment for progress bar
                if progress_task is not None:
                    m = re.search(r"Approved:\s*(\d+)", metrics)
                    approved_now = int(m.group(1)) if m else last_approved
                    if approved_now > last_approved:
                        progress.update(progress_task, completed=approved_now)
                    last_approved = approved_now

                # Parse new log lines
                lines = log_text.splitlines()
                new_lines = lines[last_len:]
                last_len = len(lines)

                for ln in new_lines:
                    if ln.startswith("========== ATTEMPT"):
                        # Extract attempt number and initialise placeholders
                        m = re.search(r"ATTEMPT\s+(\d+)", ln)
                        if m:
                            current_attempt = int(m.group(1))
                            _ensure_attempt(current_attempt)
                            attempt_starts[current_attempt] = time.time()
                    elif ln.startswith("[SEED]"):
                        _ensure_attempt(current_attempt)
                        if "prompt" in ln.lower():
                            m = re.search(r"seed (\d+)", ln)
                            if m:
                                attempts[current_attempt]["PromptSeed"] = m.group(1)
                        else:
                            m = re.search(r"seed (\d+)", ln)
                            if m:
                                attempts[current_attempt]["ImageSeed"] = m.group(1)
                    elif ln.startswith("[CHECK] Results:"):
                        _ensure_attempt(current_attempt)
                        # Start collecting multiline JSON
                        collecting_check[current_attempt] = [ln.split("Results:",1)[1]]
                    elif ln.startswith("[WF2] Image generated:"):
                        _ensure_attempt(current_attempt)
                        path = ln.split(":", 1)[1].strip()
                        # Try to resolve final saved filename
                        try:
                            final_dir_arg = values[8] if len(values) > 8 and values[8] else str(Path.home()/"ComfyUI"/"final")
                            fdir = Path(final_dir_arg).expanduser()
                            latest = max(fdir.glob("*.png"), key=lambda p: p.stat().st_mtime)
                            attempts[current_attempt]["SavedAs"] = latest.name
                            if current_attempt in attempt_starts:
                                attempts[current_attempt]["Duration"] = time.time() - attempt_starts[current_attempt]
                        except Exception:
                            attempts[current_attempt]["SavedAs"] = os.path.basename(path)
                        attempts[current_attempt]["Status"] = "OK"
                    elif ln.startswith("[WF1] Running Workflow1"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "WF1"
                    elif ln.startswith("[CHECK] Running external checks"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "Check"
                    elif ln.startswith("[WF2] Running Workflow2"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "WF2"
                        # Checks passed successfully
                        attempts[current_attempt]["Checks"] = "OK"
                    elif "pipeline stops" in ln.lower() or "error" in ln.lower():
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "Failed"
                        attempts[current_attempt]["Checks"] = "Failed"

                    # ---------------- collect check JSON -------------------
                    if collecting_check.get(current_attempt):
                        collecting_check[current_attempt].append(ln)
                        if "}" in ln:
                            blob = "\n".join(collecting_check[current_attempt])
                            try:
                                res = json.loads(blob)
                                chk_parts = []
                                if res.get("is_nsfw") is not None:
                                    chk_parts.append("NSFW" if res["is_nsfw"] else "SFW")
                                if "face_count" in res and res["face_count"] >= 0:
                                    chk_parts.append(f"faces:{res['face_count']}")
                                if res.get("is_partial_face") is not None:
                                    chk_parts.append("partial" if res["is_partial_face"] else "full")
                                attempts[current_attempt]["Checks"] = ", ".join(chk_parts) if chk_parts else "-"
                            except Exception:
                                attempts[current_attempt]["Checks"] = "-"
                            collecting_check.pop(current_attempt, None)

                # Update renderable in place ---------------------------------
                renderable = _render_group()
                live.update(renderable, refresh=True)

        except KeyboardInterrupt:
            console.print("[bold red]\n[CANCEL][/bold red] KeyboardInterrupt received. Exiting…")
            gui.CANCEL_REQUESTED = True
            try:
                for _ in gen:
                    pass
            except Exception:
                pass

    console.print("[bold green]Done[/bold green]")


def _stream_pipeline_basic(values: List[Any], expected_runs: int | None):
    from tqdm import tqdm

    gen = gui.run_pipeline_gui(*values)

    pbar = tqdm(total=expected_runs, unit="img", desc="Approved", leave=True) if expected_runs else None

    last_len = 0
    last_approved = 0

    try:
        for step in gen:
            log_text, pending, status, metrics = step[1], step[4], step[5], step[6]
            if pbar is not None:
                m = re.search(r"Approved:\s*(\d+)", metrics)
                approved_now = int(m.group(1)) if m else last_approved
                if approved_now > last_approved:
                    pbar.update(approved_now - last_approved)
                last_approved = approved_now

            lines = log_text.splitlines()
            new_line = lines[-1] if lines else ""
            print(f"[{status}] {pending} – {new_line}")

    except KeyboardInterrupt:
        print("[CANCEL] KeyboardInterrupt received. Exiting…")
        gui.CANCEL_REQUESTED = True
        try:
            for _ in gen:
                pass
        except Exception:
            pass
    finally:
        if pbar:
            pbar.close()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run ComfyUI preset from command line.")
    parser.add_argument("preset", help="Path or name of the preset JSON file")
    parser.add_argument(
        "runs",
        nargs="?",
        type=int,
        help="Desired number of successful final images (overrides stored preset)",
    )
    args = parser.parse_args(argv)

    # 1. Load stored GUI component values -----------------------------------
    values = _load_preset_values(args.preset)

    # 2. Optionally override the Batch Runs value ---------------------------
    if args.runs is not None:
        if BATCH_RUNS_INDEX >= len(values):
            # Extend list if batch index missing
            values += [None] * (BATCH_RUNS_INDEX - len(values) + 1)
        values[BATCH_RUNS_INDEX] = int(args.runs)

    # 3. Trim / pad to expected length --------------------------------------
    values = _normalize_values(values)

    # Determine expected runs (for progress bar)
    endless_mode = bool(values[10]) if len(values) > 10 else False
    expected_runs = None if endless_mode else int(values[9]) if len(values) > 9 else None

    # 4. Execute the pipeline ----------------------------------------------
    print("[INFO] Starting pipeline… Press Ctrl+C to cancel.")
    _stream_pipeline(values, expected_runs)
    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    main() 