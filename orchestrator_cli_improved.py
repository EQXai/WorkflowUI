#!/usr/bin/env python3
"""Command-line interface to run ComfyUI workflows using a saved GUI preset.

IMPROVED VERSION with enhanced interface, additional information, and real-time updates.

Usage examples
--------------
Run a preset exactly as saved:
    python orchestrator_cli_improved.py MyPreset.json

Run the same preset but aim for 55 successful final images:
    python orchestrator_cli_improved.py MyPreset.json 55

The script locates presets relative to the *save_presets* folder if the given
path does not exist. It then loads the stored list of component values (the
same list Gradio stores) and forwards them to *run_pipeline_gui* from
*orchestrator_gui.py*.  A comprehensive console progress is printed while the
pipeline executes.  No Gradio UI is launched.
"""
from __future__ import annotations

import argparse
import json
import sys
import re
import os
from pathlib import Path
from typing import List, Any, Dict, Optional
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Local modules – orchestrator_gui contains the heavy lifting.
import orchestrator_gui as gui  # noqa: E402

# Attempt to import rich for fancy CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
# Data structures for improved tracking
# -----------------------------------------------------------------------------

@dataclass
class AttemptInfo:
    """Information about a single attempt."""
    id: int
    status: str = "Queued"
    prompt_seed: str = "-"
    image_seed: str = "-"
    checks_info: str = "Pending"
    saved_as: str = ""
    duration: Optional[float] = None
    start_time: Optional[float] = None
    wf1_start: Optional[float] = None
    wf1_duration: Optional[float] = None
    check_start: Optional[float] = None
    check_duration: Optional[float] = None
    wf2_start: Optional[float] = None
    wf2_duration: Optional[float] = None
    error_message: str = ""

@dataclass
class GlobalMetrics:
    """Global execution metrics."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    in_progress_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    approved_images: int = 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100

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
        sys.exit(f"[ERROR] Could not read preset file \'{p}\': {exc}")


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


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds."""
    if seconds is None:
        return "-"
    return f"{seconds:.1f}s"


def _format_checks_info(checks_str: str) -> str:
    """Format checks information in a more user-friendly way."""
    if not checks_str or checks_str in ["-", "Pending", "Failed"]:
        return checks_str
    
    # Parse the checks string and make it more readable
    parts = checks_str.split(", ")
    formatted_parts = []
    
    for part in parts:
        if part == "NSFW":
            formatted_parts.append("🔞 NSFW")
        elif part == "SFW":
            formatted_parts.append("✅ SFW")
        elif part.startswith("faces:"):
            count = part.split(":")[1]
            formatted_parts.append(f"👤 {count} face(s)")
        elif part == "partial":
            formatted_parts.append("⚠️ Partial face")
        elif part == "full":
            formatted_parts.append("✅ Full face")
    
    return " | ".join(formatted_parts) if formatted_parts else checks_str


def _stream_pipeline_rich(values: List[Any], expected_runs: int | None, preset_name: str):
    """Run *run_pipeline_gui* and display a Rich Live interface with comprehensive progress."""
    
    console = Console()
    
    # Display preset information at startup
    preset_info = Panel(
        f"[bold]Preset:[/bold] {preset_name}\n"
        f"[bold]Expected Images:[/bold] {expected_runs if expected_runs else 'Unlimited'}\n"
        f"[bold]Started:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="🚀 Pipeline Configuration",
        border_style="blue"
    )
    console.print(preset_info)
    console.print()

    gen = gui.run_pipeline_gui(*values)

    # Data storage
    attempts: Dict[int, AttemptInfo] = {}
    current_attempt = 0
    metrics = GlobalMetrics()
    collecting_check: Dict[int, List[str]] = {}

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
        progress_task = progress.add_task("✅ Approved Images", total=expected_runs)

    def _build_attempts_table() -> Table:
        """Build the main attempts table."""
        table = Table(title="🔄 Pipeline Attempts", expand=True, show_header=True)
        table.add_column("ID", justify="right", style="cyan", width=2)
        table.add_column("Status", justify="center", width=7)
        table.add_column("Prompt Seed", justify="right", style="yellow", width=12)
        table.add_column("Image Seed", justify="right", style="yellow", width=12)
        table.add_column("Checks", justify="left", width=12)
        table.add_column("Duration", justify="right", style="green", width=7)
        table.add_column("Saved As", justify="left", style="white", overflow="ellipsis", max_width=30)
        
        for attempt_id in sorted(attempts.keys()):
            attempt = attempts[attempt_id]
            
            # Color-code status
            status_colors = {
                "Queued": "grey50",
                "WF1": "yellow",
                "Check": "cyan",
                "WF2": "bright_blue",
                "OK": "green",
                "Failed": "red",
            }
            status_color = status_colors.get(attempt.status, "white")
            colored_status = f"[{status_color}]{attempt.status}[/{status_color}]"
            
            # Format checks info
            formatted_checks = _format_checks_info(attempt.checks_info)
            
            table.add_row(
                str(attempt.id),
                colored_status,
                attempt.prompt_seed,
                attempt.image_seed,
                formatted_checks,
                _format_duration(attempt.duration),
                attempt.saved_as,
            )
        
        return table

    def _build_metrics_panel() -> Panel:
        """Build the metrics panel."""
        success_rate = f"{metrics.success_rate:.1f}%" if metrics.total_attempts > 0 else "0%"
        
        # Calculate average durations
        successful_durations = [a.duration for a in attempts.values() 
                              if a.status == "OK" and a.duration is not None]
        avg_duration = sum(successful_durations) / len(successful_durations) if successful_durations else 0
        
        # Estimate time remaining based on average duration and expected runs
        if expected_runs is not None and metrics.successful_attempts > 0 and avg_duration > 0:
            remaining = max(expected_runs - metrics.approved_images, 0)
            eta_seconds = remaining * avg_duration
            eta_text = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_text = "-:--:--"
        
        metrics_text = (
            f"[bold]Total Attempts:[/bold] {metrics.total_attempts}\n"
            f"[bold green]Successful:[/bold green] {metrics.successful_attempts}\n"
            f"[bold red]Failed:[/bold red] {metrics.failed_attempts}\n"
            f"[bold yellow]In Progress:[/bold yellow] {metrics.in_progress_attempts}\n"
            f"[bold]Success Rate:[/bold] {success_rate}\n"
            f"[bold]Avg Duration:[/bold] {_format_duration(avg_duration)}\n"
            f"[bold]Total Time:[/bold] {_format_duration(metrics.elapsed_time)}\n"
            f"[bold]ETA:[/bold] {eta_text}"
        )
        
        return Panel(
            metrics_text,
            title="📊 Execution Metrics",
            border_style="green"
        )

    def _build_layout():
        """Build the complete layout."""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="metrics", size=10),
            Layout(name="attempts")
        )
        
        # Fill sections
        if progress_task is not None:
            layout["progress"].update(progress)
        else:
            layout["progress"].update(Panel("Running indefinitely...", border_style="blue"))
            
        layout["metrics"].update(_build_metrics_panel())
        layout["attempts"].update(_build_attempts_table())
        
        return layout

    def _ensure_attempt(idx: int):
        """Ensure attempt exists in tracking."""
        if idx not in attempts:
            attempts[idx] = AttemptInfo(id=idx)
            metrics.total_attempts = max(metrics.total_attempts, idx)

    def _update_metrics():
        """Update global metrics based on current attempts."""
        metrics.successful_attempts = sum(1 for a in attempts.values() if a.status == "OK")
        metrics.failed_attempts = sum(1 for a in attempts.values() if a.status == "Failed")
        metrics.in_progress_attempts = sum(1 for a in attempts.values() 
                                         if a.status in ["WF1", "Check", "WF2"])

    # Main execution loop
    with Live(_build_layout(), console=console, refresh_per_second=2, screen=False) as live:
        try:
            last_len = 0
            last_approved = 0

            for step in gen:
                log_text, pending, status, step_metrics = step[1], step[4], step[5], step[6]

                # Update approved count for progress bar
                if progress_task is not None:
                    m = re.search(r"Approved:\s*(\d+)", step_metrics)
                    approved_now = int(m.group(1)) if m else last_approved
                    if approved_now > last_approved:
                        progress.update(progress_task, completed=approved_now)
                        metrics.approved_images = approved_now
                    last_approved = approved_now

                # Parse new log lines
                lines = log_text.splitlines()
                new_lines = lines[last_len:]
                last_len = len(lines)

                for ln in new_lines:
                    current_time = time.time()
                    
                    if ln.startswith("========== ATTEMPT"):
                        # Extract attempt number and initialize
                        m = re.search(r"ATTEMPT\s+(\d+)", ln)
                        if m:
                            current_attempt = int(m.group(1))
                            _ensure_attempt(current_attempt)
                            attempts[current_attempt].start_time = current_time
                            
                    elif ln.startswith("[SEED]"):
                        _ensure_attempt(current_attempt)
                        lower_ln = ln.lower()
                        # Extract the numeric value found AFTER the colon – this is
                        # the actual seed we are interested in.
                        m = re.search(r":\s*(\d+)", ln)
                        if not m:
                            # Nothing to capture → skip
                            continue
                        seed_val = m.group(1)

                        # Distinguish the seed type based on the descriptive text
                        if "prompt loader" in lower_ln:
                            # Prompt seed (node 190)
                            attempts[current_attempt].prompt_seed = seed_val
                        elif "seed everywhere" in lower_ln:
                            # Image seed coming from the *Seed Everywhere* node (189)
                            attempts[current_attempt].image_seed = seed_val
                        # Ignore the remaining noise_seed lines – they are not
                        # displayed in the main table and would otherwise override
                        # the image seed captured above.
                        
                    elif ln.startswith("[CHECK] Results:"):
                        _ensure_attempt(current_attempt)
                        collecting_check[current_attempt] = [ln.split("Results:", 1)[1]]
                        
                    elif ln.startswith("[WF2] Image generated:"):
                        _ensure_attempt(current_attempt)
                        path = ln.split(":", 1)[1].strip()
                        
                        # Try to resolve final saved filename
                        try:
                            final_dir_arg = values[8] if len(values) > 8 and values[8] else str(Path.home()/"ComfyUI"/"final")
                            fdir = Path(final_dir_arg).expanduser()
                            latest = max(fdir.glob("*.png"), key=lambda p: p.stat().st_mtime)
                            attempts[current_attempt].saved_as = latest.name
                        except Exception:
                            attempts[current_attempt].saved_as = os.path.basename(path)
                            
                        attempts[current_attempt].status = "OK"
                        if attempts[current_attempt].start_time:
                            attempts[current_attempt].duration = current_time - attempts[current_attempt].start_time
                            
                    elif ln.startswith("[WF1] Running Workflow1"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt].status = "WF1"
                        attempts[current_attempt].wf1_start = current_time
                        
                    elif ln.startswith("[CHECK] Running external checks"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt].status = "Check"
                        if attempts[current_attempt].wf1_start:
                            attempts[current_attempt].wf1_duration = current_time - attempts[current_attempt].wf1_start
                        attempts[current_attempt].check_start = current_time
                        
                    elif ln.startswith("[WF2] Running Workflow2"):
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt].status = "WF2"
                        attempts[current_attempt].checks_info = "OK"
                        if attempts[current_attempt].check_start:
                            attempts[current_attempt].check_duration = current_time - attempts[current_attempt].check_start
                        attempts[current_attempt].wf2_start = current_time
                        
                    elif "pipeline stops" in ln.lower() or "error" in ln.lower():
                        _ensure_attempt(current_attempt)
                        attempts[current_attempt].status = "Failed"
                        attempts[current_attempt].checks_info = "Failed"
                        attempts[current_attempt].error_message = ln

                    # Collect check JSON
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
                                attempts[current_attempt].checks_info = ", ".join(chk_parts) if chk_parts else "-"
                            except Exception:
                                attempts[current_attempt].checks_info = "-"
                            collecting_check.pop(current_attempt, None)

                # Update metrics and refresh display
                _update_metrics()
                live.update(_build_layout(), refresh=True)

        except KeyboardInterrupt:
            console.print("\n[bold red]⚠️  KeyboardInterrupt received. Stopping pipeline...[/bold red]")
            gui.CANCEL_REQUESTED = True
            try:
                for _ in gen:
                    pass
            except Exception:
                pass

    # Final summary
    _print_final_summary(console, attempts, metrics, preset_name)


def _print_final_summary(console: Console, attempts: Dict[int, AttemptInfo], 
                        metrics: GlobalMetrics, preset_name: str):
    """Print a comprehensive final summary."""
    
    successful_files = [a.saved_as for a in attempts.values() if a.status == "OK" and a.saved_as]
    
    summary_text = (
        f"[bold]Preset:[/bold] {preset_name}\n"
        f"[bold]Total Execution Time:[/bold] {_format_duration(metrics.elapsed_time)}\n"
        f"[bold]Total Attempts:[/bold] {metrics.total_attempts}\n"
        f"[bold green]Successful Images:[/bold green] {metrics.successful_attempts}\n"
        f"[bold red]Failed Attempts:[/bold red] {metrics.failed_attempts}\n"
        f"[bold]Success Rate:[/bold] {metrics.success_rate:.1f}%\n"
    )
    
    if successful_files:
        summary_text += f"\n[bold]Generated Files:[/bold]\n"
        for file in successful_files[:10]:  # Show first 10 files
            summary_text += f"  • {file}\n"
        if len(successful_files) > 10:
            summary_text += f"  ... and {len(successful_files) - 10} more files\n"
    
    summary_panel = Panel(
        summary_text,
        title="🎉 Execution Summary",
        border_style="green"
    )
    
    console.print("\n")
    console.print(summary_panel)


def _stream_pipeline_basic(values: List[Any], expected_runs: int | None, preset_name: str):
    """Fallback implementation when rich is not available."""
    
    print(f"🚀 Starting pipeline with preset: {preset_name}")
    print(f"📊 Expected images: {expected_runs if expected_runs else 'Unlimited'}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=expected_runs, unit="img", desc="Approved", leave=True) if expected_runs else None
    except ImportError:
        pbar = None

    gen = gui.run_pipeline_gui(*values)
    
    start_time = time.time()
    last_approved = 0
    attempt_count = 0
    successful_count = 0

    try:
        for step in gen:
            log_text, pending, status, step_metrics = step[1], step[4], step[5], step[6]
            
            # Update progress bar if available
            if pbar is not None:
                m = re.search(r"Approved:\s*(\d+)", step_metrics)
                approved_now = int(m.group(1)) if m else last_approved
                if approved_now > last_approved:
                    pbar.update(approved_now - last_approved)
                    successful_count = approved_now
                last_approved = approved_now

            # Count attempts
            lines = log_text.splitlines()
            for line in lines:
                if line.startswith("========== ATTEMPT"):
                    attempt_count += 1
                    print(f"🔄 Starting attempt {attempt_count}")
                elif line.startswith("[WF2] Image generated:"):
                    print(f"✅ Image generated successfully")

            # Show current status
            elapsed = time.time() - start_time
            print(f"[{_format_duration(elapsed)}] {status} - {pending}")

    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt received. Stopping pipeline...")
        gui.CANCEL_REQUESTED = True
        try:
            for _ in gen:
                pass
        except Exception:
                pass
    finally:
        if pbar:
            pbar.close()
        
        # Print final summary
        total_time = time.time() - start_time
        success_rate = (successful_count / attempt_count * 100) if attempt_count > 0 else 0
        
        print("\n" + "=" * 60)
        print("🎉 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Preset: {preset_name}")
        print(f"Total Time: {_format_duration(total_time)}")
        print(f"Total Attempts: {attempt_count}")
        print(f"Successful Images: {successful_count}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 60)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run ComfyUI preset from command line with enhanced interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s MyPreset.json              # Run preset as saved
  %(prog)s MyPreset.json 50           # Override to generate 50 images
  %(prog)s presets/custom.json        # Use specific path
        """
    )
    parser.add_argument("preset", help="Path or name of the preset JSON file")
    parser.add_argument(
        "runs",
        nargs="?",
        type=int,
        help="Desired number of successful final images (overrides stored preset)",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Force basic text output even if rich is available"
    )
    args = parser.parse_args(argv)

    # 1. Load stored GUI component values
    values = _load_preset_values(args.preset)
    preset_name = Path(args.preset).stem

    # 2. Optionally override the Batch Runs value
    if args.runs is not None:
        if BATCH_RUNS_INDEX >= len(values):
            # Extend list if batch index missing
            values += [None] * (BATCH_RUNS_INDEX - len(values) + 1)
        values[BATCH_RUNS_INDEX] = int(args.runs)

    # 3. Normalize values for headless execution
    values = _normalize_values(values)

    # 4. Run the pipeline with appropriate interface
    if RICH_AVAILABLE and not args.no_rich:
        _stream_pipeline_rich(values, args.runs, preset_name)
    else:
        if not RICH_AVAILABLE:
            print("⚠️  'rich' library not available. Using basic interface.")
        _stream_pipeline_basic(values, args.runs, preset_name)


if __name__ == "__main__":
    main()

