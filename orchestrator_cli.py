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
import re as _re_cli
from collections import deque
import threading

# Local modules – orchestrator_gui contains the heavy lifting.
import orchestrator_gui as gui  # noqa: E402

from datetime import datetime

# Attempt to import rich for fancy CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.panel import Panel
    from rich.rule import Rule
except ImportError:
    Console = None  # Fallback if rich not installed

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PRESET_DIR = THIS_DIR / "save_presets"
BATCH_RUNS_INDEX = 10  # Index actualizado tras añadir wf1_mode al inicio de ALL_COMPONENTS

# Number of fixed parameters before dynamic_values in run_pipeline_gui
_FIXED_COUNT = 20
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
    ("190", "file_path", str),
    ("293", "base_dir", str),
    ("293", "positive_joiner", str),
    ("293", "negative_joiner", str),
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

    # Asegurar que la lista contiene al menos los parámetros fijos + categorías
    if len(values) < prefix_len:
        values += [None] * (prefix_len - len(values))

    sanitized = values[:prefix_len] + overrides_seg + lora_tail
    return sanitized


def _stream_pipeline(values: List[Any], expected_runs: int | None, wf1_mode: str):
    """Run *run_pipeline_gui* and display a Rich Live table with progress."""

    if Console is None:
        print("[WARN] 'rich' not installed. Falling back to basic logging.")
        return _stream_pipeline_basic(values, expected_runs)

    console = Console()
    try:
        console.clear()
    except Exception:
        # Fallback ANSI clear
        print("\033c", end="")

    # Extract Destination Folder for Final Images (index 9 after normalization)
    final_dir_str = str(values[9]) if len(values) > 9 and values[9] else "-"

    gen = gui.run_pipeline_gui(*values)

    # Data storage per attempt
    attempts: dict[int, dict[str, str]] = {}
    attempt_starts: dict[int, float] = {}
    current_attempt = 0
    # Track JSON collection per attempt
    collecting_check: dict[int, list[str]] = {}
    recent_log = deque(maxlen=50)

    # Progress bar setup
    from rich.progress import ProgressColumn
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )

    progress_task = None
    if expected_runs:
        progress_task = progress.add_task("Approved", total=expected_runs)

    # Start time for statistics (needs to be available before first _render_group call)
    start_time_cli = time.time()

    def _build_table() -> Table:
        saved_width = max(12, int(console.size.width * 0.2))  # 20 % ancho máx.
        title_top = "╔═ PIPELINE RUNS ═╗"
        title_sub = f"({wf1_mode})"
        title_markup = (
            f"[bold bright_cyan]{title_top}[/bold bright_cyan]\n"
            f"[white]{title_sub}[/white]"
        )
        table = Table(title=title_markup, title_justify="center", expand=True)
        table.add_column("Attempt", justify="right")
        table.add_column("Status")
        table.add_column("PromptSeed", justify="right")
        table.add_column("ImageSeed", justify="right")
        table.add_column("Checks")
        table.add_column("Saved As", max_width=saved_width, overflow="ellipsis")
        return table

    def _render_group():
        tbl = _build_table()

        # Output dir panel -------------------------------------------------
        from rich.panel import Panel as _Pnl
        from rich.text import Text as _Txt
        out_panel = _Pnl(_Txt.from_markup(f"[bold]Final dir:[/bold] {final_dir_str}"), expand=False, border_style="grey50")

        # ---- statistics line ---
        attempts_total = len(attempts)
        accepted_cnt = len([v for v in attempts.values() if v.get("Status") == "OK"])
        rejected_cnt = len([v for v in attempts.values() if v.get("Status") in ("NOT VALID", "Failed")])
        target_txt = str(expected_runs) if expected_runs is not None else "∞"
        remaining = "∞" if expected_runs is None else max(0, expected_runs - accepted_cnt)
        elapsed_sec = int(time.time() - start_time_cli)
        h, rem = divmod(elapsed_sec, 3600)
        m, s = divmod(rem, 60)
        elapsed_txt = f"{h:02d}:{m:02d}:{s:02d}"
        eta_txt = "-"
        if expected_runs is not None and accepted_cnt > 0:
            rate = elapsed_sec / accepted_cnt
            eta_sec = int(rate * remaining)
            hh, rr = divmod(eta_sec, 3600)
            mm, ss = divmod(rr, 60)
            eta_txt = f"{hh:02d}:{mm:02d}:{ss:02d}"

        # ---- average time calculation ----
        durations = [v.get("Duration") for v in attempts.values() if v.get("Status") == "OK" and v.get("Duration")]
        avg_txt = "-" if not durations else f"{sum(durations)/len(durations):.1f}s"

        stats_line = (
            f"[bold]Attempts[/bold]: {attempts_total}   "
            f"[bold]Accepted[/bold]: {accepted_cnt}   "
            f"[bold]Rejected[/bold]: {rejected_cnt}   "
            f"[bold]Objective[/bold]: {target_txt}   "
            f"[bold]Remaining[/bold]: {remaining}   "
            f"[bold]Elapsed[/bold]: {elapsed_txt}   "
            f"[bold]ETA[/bold]: {eta_txt}   "
            f"[bold]Avg/Image[/bold]: {avg_txt}"
        )

        from rich.console import Group
        from rich.text import Text
        stats_render = Text.from_markup(stats_line)
        stats_panel = Panel(stats_render, expand=False, border_style="grey50")

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

        if progress_task is not None:
            separator = Rule(style="grey50")
            return Group(separator, progress, out_panel, tbl, stats_panel)
        return Group(out_panel, tbl, stats_panel)

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

    # Background thread to refresh live display every second so stats update even during long operations
    stop_refresh = threading.Event()

    def _auto_refresh():
        while not stop_refresh.is_set():
            time.sleep(1)
            live.refresh()

    with Live(renderable, console=console, refresh_per_second=4, screen=False) as live:
        refresher = threading.Thread(target=_auto_refresh, daemon=True)
        refresher.start()
        try:
            last_len = 0
            last_approved = 0
            last_seed_len = 0  # Track processed length of seeds log
            last_refresh = 0.0

            for step in gen:
                log_text = step[1] if len(step) > 1 else ""
                seeds_text = step[2] if len(step) > 2 else ""
                pending = step[5] if len(step) > 5 else ""
                status = step[6] if len(step) > 6 else ""
                metrics = step[7] if len(step) > 7 and step[7] is not None else ""

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
                            new_attempt = int(m.group(1))
                            if new_attempt != current_attempt and current_attempt != 0:
                                # Keep cumulative log (no clear)
                                pass
                            current_attempt = new_attempt
                            _ensure_attempt(current_attempt)
                            attempt_starts[current_attempt] = time.time()
                    elif ln.startswith("[SEED]"):
                        _ensure_attempt(current_attempt)
                        # Extract the numeric value that appears after the last ':'
                        m = re.search(r":\s*(\d+)", ln)
                        if not m:
                            continue  # Skip if no number found
                        seed_val = m.group(1)

                        ln_lower = ln.lower()
                        if "prompt" in ln_lower:
                            # Prompt loader seed (node 190)
                            attempts[current_attempt]["PromptSeed"] = seed_val
                        elif (
                            ("seed_ksampler_1" in ln_lower or "seed_ksampler 1" in ln_lower)
                            or ("seed_ksampler_2" in ln_lower or "seed_ksampler 2" in ln_lower)
                            or "seed everywhere" in ln_lower  # compat
                        ) and not attempts[current_attempt]["ImageSeed"].isdigit():
                            # First occurrence corresponds to image seed (nodes 189/285) or legacy Seed Everywhere
                            attempts[current_attempt]["ImageSeed"] = seed_val
                    elif ln.startswith("[CHECK]") and (
                        "flagged as NSFW" in ln
                        or "More than one face" in ln
                        or "Partial face" in ln
                    ):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "NOT VALID"
                        attempts[current_attempt]["Checks"] = "NO"
                        # Use the part of the line after '[CHECK]' as reason (trimmed)
                        reason = ln.split("[CHECK]", 1)[1].strip()
                        # Remove verbose suffixes
                        reason = _re_cli.sub(r"\.\s*Pipeline stops\.*", "", reason, flags=_re_cli.I)
                        reason = _re_cli.sub(r"\s*and stop option is active\.*", "", reason, flags=_re_cli.I)
                        reason = reason.strip()
                        # Only set if not already set to NSFW-<cat>
                        if not attempts[current_attempt].get("SavedAs", "").startswith("NSFW-"):
                            attempts[current_attempt]["SavedAs"] = reason
                    elif ln.startswith("[CHECK] Running external checks"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "Check"
                    elif ln.startswith("[CHECK] Results:"):
                        _ensure_attempt(current_attempt)
                        # Start collecting multiline JSON
                        collecting_check[current_attempt] = [ln.split("Results:",1)[1]]
                    elif ln.startswith("[FINAL] Image saved as:"):
                        _ensure_attempt(current_attempt)
                        saved_name = os.path.basename(ln.split(":", 1)[1].strip())
                        attempts[current_attempt]["SavedAs"] = saved_name
                        if current_attempt in attempt_starts and attempts[current_attempt].get("Duration") is None:
                            attempts[current_attempt]["Duration"] = time.time() - attempt_starts[current_attempt]
                        attempts[current_attempt]["Status"] = "OK"
                    elif ln.startswith("[WF1] Running Workflow1"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "WF1"
                    elif "pipeline stops" in ln.lower():
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "NOT VALID"
                        attempts[current_attempt]["Checks"] = "NO"
                        if not attempts[current_attempt]["SavedAs"] or attempts[current_attempt]["SavedAs"] == "None":
                            attempts[current_attempt]["SavedAs"] = "Checks failed"
                    elif "error" in ln.lower():
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "Failed"
                        attempts[current_attempt]["Checks"] = "Failed"
                    elif ln.startswith("[WF2] Running Workflow2"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt]["Status"] = "WF2"
                        # If checks were pending, mark them as OK now
                        if attempts[current_attempt]["Checks"] in ("Pending", "-", "NO"):
                            attempts[current_attempt]["Checks"] = "OK"

                    # ---------------- collect check JSON -------------------
                    if collecting_check.get(current_attempt):
                        collecting_check[current_attempt].append(ln)
                        if "}" in ln:
                            blob = "\n".join(collecting_check[current_attempt])
                            try:
                                res = json.loads(blob)
                                chk_parts = []
                                if res.get("is_nsfw") is not None:
                                    if res["is_nsfw"]:
                                        cat = res.get("nsfw_category", "?")
                                        chk_parts.append(f"NSFW-{cat}")
                                        attempts[current_attempt]["SavedAs"] = f"NSFW-{cat}"
                                    else:
                                        chk_parts.append("SFW")
                                if "face_count" in res and res["face_count"] >= 0:
                                    chk_parts.append(f"faces:{res['face_count']}")
                                if res.get("is_partial_face") is not None:
                                    chk_parts.append("partial" if res["is_partial_face"] else "full")
                                attempts[current_attempt]["Checks"] = ", ".join(chk_parts) if chk_parts else "-"
                            except Exception:
                                attempts[current_attempt]["Checks"] = "-"
                            collecting_check.pop(current_attempt, None)

                # -------------- Parse Seeds Log (panel) -----------------------
                seeds_lines = seeds_text.splitlines()
                new_seed_lines = seeds_lines[last_seed_len:]
                last_seed_len = len(seeds_lines)

                seed_attempt = None
                for s_ln in new_seed_lines:
                    if s_ln.startswith("Attempt"):
                        m_at = re.search(r"Attempt\s+(\d+)", s_ln)
                        if m_at:
                            seed_attempt = int(m_at.group(1))
                            _ensure_attempt(seed_attempt)
                        continue

                    if seed_attempt is None:
                        continue

                    # PromptSeed detection depending on mode
                    if wf1_mode.lower().startswith("prompt") and "PromptConcat seed" in s_ln:
                        m = re.search(r":\s*(\d+)", s_ln)
                        if m:
                            attempts[seed_attempt]["PromptSeed"] = m.group(1)
                    elif not wf1_mode.lower().startswith("prompt") and "Prompt loader seed" in s_ln:
                        m = re.search(r":\s*(\d+)", s_ln)
                        if m:
                            attempts[seed_attempt]["PromptSeed"] = m.group(1)

                    # Image seed always from Seed_KSampler_1 (189)
                    if "Seed_KSampler_1" in s_ln:
                        m = re.search(r":\s*(\d+)", s_ln)
                        if m:
                            attempts[seed_attempt]["ImageSeed"] = m.group(1)

                    # Append every processed line to rolling log
                    recent_log.append(s_ln)

                # Update renderable in place after processing current batch
                live.update(_render_group(), refresh=True)

        except KeyboardInterrupt:
            console.print("[bold red]\n[CANCEL][/bold red] KeyboardInterrupt received. Exiting…")
            gui.CANCEL_REQUESTED = True
            try:
                for _ in gen:
                    pass
            except Exception:
                pass
        finally:
            if progress_task is not None:
                progress.stop_task(progress_task)
            stop_refresh.set()
            refresher.join(timeout=1)

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
        help="Number of successful images to generate or 'nolimit' for endless mode",
    )
    args = parser.parse_args(argv)

    # 1. Load stored GUI component values -----------------------------------
    values = _load_preset_values(args.preset)

    # 2. Optionally override the Batch Runs value ---------------------------
    if args.runs is not None and str(args.runs).lower() != "nolimit":
        if BATCH_RUNS_INDEX >= len(values):
            # Extend list if batch index missing
            values += [None] * (BATCH_RUNS_INDEX - len(values) + 1)
        try:
            values[BATCH_RUNS_INDEX] = int(args.runs)
        except ValueError:
            sys.exit("[ERROR] 'runs' must be an integer or 'nolimit'.")

    # Handle nolimit: force endless_until_cancel flag (index 11)
    if str(args.runs).lower() == "nolimit":
        # Ensure list is long enough
        endless_idx = 11
        if endless_idx >= len(values):
            values += [None] * (endless_idx - len(values) + 1)
        values[endless_idx] = True

    # 3. Trim / pad to expected length --------------------------------------
    values = _normalize_values(values)

    # Clear terminal before starting rich interface
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    except Exception:
        print("\033[2J\033[H", end="")

    # Determine WF1 mode from first value (index 0)
    wf1_mode = values[0] if values else "Unknown"
    print(f"[INFO] Starting pipeline (WF1 Mode: {wf1_mode})… Press Ctrl+C to cancel.")

    # Determine expected runs (for progress bar)
    endless_mode = bool(values[11]) if len(values) > 11 else False

    if str(args.runs).lower() == "nolimit":
        expected_runs = None
    elif args.runs is not None:
        expected_runs = int(args.runs)
    else:
        expected_runs = None if endless_mode else None

    # 4. Execute the pipeline ----------------------------------------------
    _stream_pipeline(values, expected_runs, wf1_mode)
    print("[INFO] Pipeline finished.")


if __name__ == "__main__":
    main() 