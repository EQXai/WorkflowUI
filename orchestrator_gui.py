import os
import time
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
from PIL import Image
import shutil
import tempfile
import numpy as np  # NEW: for beep sound generation

# Local modules
import orchestrator as oc  # our previously created orchestrator.py

# Constants
DEFAULT_OUTPUT_DIR = Path(oc.DEFAULT_OUTPUT_DIR)

# Directory where preset JSON files will be stored
PRESET_DIR = Path(__file__).resolve().parent / "save_presets"
PRESET_DIR.mkdir(parents=True, exist_ok=True)

# Extra constants for advanced check configuration
from run_checks import CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_SEED_VALUE = 10_000_000  # Upper limit for any seed value

# NEW: Maximum number of images kept in the on-screen gallery
MAX_GALLERY_IMAGES = 20  # Only the last 20 approved images will be shown

# -------------------------------------------------------------
# Dynamic form helpers for workflow editing
# -------------------------------------------------------------

# Will be populated during GUI build so that run_pipeline_gui can
# access the mapping information.
WF1_MAPPING: list[tuple[str, str, type]] = []  # (node_id, input_key, original_type)
WF2_MAPPING: list[tuple[str, str, type]] = []

# Flag used to signal cancellation from the GUI
CANCEL_REQUESTED = False

# External seed counter that persists across pipeline invocations while the
# application is running. It is only used when the user selects the
# "Incremental" seed mode.
GLOBAL_SEED_COUNTER: int | None = None

# Global counters for statistics
GLOBAL_APPROVED_COUNT = 0
GLOBAL_REJECTED_COUNT = 0

# Placeholder beep generator -------------------------------------------------

AUDIO_FILE = Path(__file__).resolve().parent / "audio" / "notify.mp3" # New constant

def _get_notification_sound():
    """Return a sound reference accepted by gr.Audio: filepath if exists else tuple(sr, arr)."""
    if AUDIO_FILE.exists():
        return str(AUDIO_FILE)
    # fallback beep
    duration = 0.6
    freq = 880
    sr = 44100
    t = np.linspace(0, duration, int(sr*duration), False, dtype=np.float32)
    tone = 0.25 * np.sin(2 * np.pi * freq * t)
    return sr, tone.astype(np.float32)

def _request_cancel():
    """Triggered by Cancel button: stop execution and update indicators."""
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    return (
        "Pending: 0",
        "Cancelled",
        f"Approved: {GLOBAL_APPROVED_COUNT} | Rejected: {GLOBAL_REJECTED_COUNT}",
        str(AUDIO_FILE) if AUDIO_FILE.exists() else None,
    )

# -------------------------------------------------------------
# Workflow editor builder
# -------------------------------------------------------------

def _build_workflow_editor(json_path: str, highlight_ids: list[str] | None = None):
    """Return (components, mapping) for the workflow located at *json_path*.

    If *highlight_ids* is provided, the indicated node IDs will appear first in
    the UI (useful for quick access to frequently-edited parameters).

    components: list of Gradio components created.
    mapping: list of tuples (node_id, input_key, original_type) in the same order
             as the returned components so that their values can later be mapped
             back into the JSON structure.
    """

    from gradio.components import Textbox, Checkbox, Slider

    data = json.loads(Path(json_path).read_text())
    comps: list = []
    mapping: list[tuple[str, str, type]] = []

    highlight_set = {str(i) for i in (highlight_ids or [])}

    # Helper to add UI components for a single node -------------------------
    def _add_node(node_id: str):
        node = data.get(node_id)
        if not node or not isinstance(node, dict):
            return
        inputs = node.get("inputs")
        if not inputs:
            return

        with gr.Accordion(f"{node_id} – {node.get('class_type', '')}", open=False):
            for key, val in inputs.items():
                # choose component type heuristically
                if isinstance(val, bool):
                    comp = Checkbox(label=key, value=val)
                elif isinstance(val, (int, float, str)):
                    comp = Textbox(label=key, value=str(val))
                else:
                    # For list/dict or any other complex type, show JSON string
                    comp = Textbox(label=key, value=json.dumps(val))

                comps.append(comp)
                mapping.append((node_id, key, type(val)))

    # First pass: highlighted nodes rendered inside a yellow box -----------
    if highlight_ids:
        with gr.Column(elem_classes=["highlight-box"]):
            gr.Markdown("**Highlighted nodes** (frequently edited):")
            for hid in highlight_ids:
                _add_node(str(hid))

    # Second pass: remaining nodes sorted for determinism -------------------
    def _node_sort(k):
        try:
            return int(k)
        except Exception:
            return k

    for node_id in sorted(data.keys(), key=_node_sort):
        if node_id in highlight_set:
            continue  # already added
        _add_node(node_id)

    return comps, mapping


def _apply_edits_to_workflow(original_path: str, mapping: list[tuple[str, str, type]], values: list[str]) -> str:
    """Create a temporary copy of *original_path* applying the edited *values*.

    Returns the path to the temporary JSON file.
    """
    data = json.loads(Path(original_path).read_text())

    for (node_id, key, typ), val in zip(mapping, values):
        # Convert string input back to original type when possible
        try:
            if typ is bool:
                cast_val = val if isinstance(val, bool) else val.lower() == "true"
            elif typ is int:
                cast_val = int(val)
            elif typ is float:
                cast_val = float(val)
            elif typ in (list, dict):
                # Expect JSON string representation
                cast_val = json.loads(val)
            else:
                cast_val = val
        except Exception:
            cast_val = val  # fallback to raw string

        if node_id in data and "inputs" in data[node_id]:
            data[node_id]["inputs"][key] = cast_val

    tmp_fd, tmp_path = tempfile.mkstemp(suffix="_workflow.json")
    os.close(tmp_fd)
    Path(tmp_path).write_text(json.dumps(data, indent=2))
    return tmp_path

# -----------------------------------------------------------------------------
# Custom CSS to show full portrait images without cropping (object-fit: contain)
# -----------------------------------------------------------------------------

CSS = """
+.pending-box {
    position: fixed;
    top: 10px;
    right: 10px;
    background-color: rgba(255, 255, 255, 0.9);
    border: 1px solid #ccc;
    padding: 6px 10px;
    font-weight: bold;
    z-index: 1000;
}

#results-gallery img {
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
}

/* Highlight container for featured nodes */
+.highlight-box {
    border: 3px solid #f8e71c !important; /* bright yellow */
    background-color: rgba(248, 231, 28, 0.05); /* subtle yellow tint */
    padding: 10px !important;
    margin-bottom: 12px !important;
    border-radius: 6px;
}
"""

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_img(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def run_pipeline_gui(
    do_nsfw: bool,
    do_face_count: bool,
    do_partial_face: bool,
    stop_multi_faces: bool,
    stop_partial_face: bool,
    confidence: float,
    margin: float,
    output_dir_str: str,
    final_dir_str: str,
    batch_runs: int,
    endless_until_cancel: bool,
    play_sound: bool,
    seed_mode: str,
    seed_counter_input: int | float,
    override_save_name: bool,
    save_name_base: str,
    load_prompts_directly: bool,
    prompt_list_str: str,
    characteristics_text: str,
    *dynamic_values: bool | str,
):
    """Function executed by the Gradio UI. Returns tuple of outputs."""
    global GLOBAL_APPROVED_COUNT, GLOBAL_REJECTED_COUNT

    def _metrics_str():
        return f"Approved: {GLOBAL_APPROVED_COUNT} | Rejected: {GLOBAL_REJECTED_COUNT}"

    def _pending_str(pend):
        return f"Pending: {pend}"

    # This variable will hold either None (no sound) or (sr, arr)
    sound_placeholder: str | tuple[int, np.ndarray] | None = None

    def _boxes(pend):
        return (
            _pending_str(pend),
            status_str,
            _metrics_str(),
            sound_placeholder,
        )

    status_str = "Idle"

    # Status helper ---------------------------------------------------------
    global CANCEL_REQUESTED, GLOBAL_SEED_COUNTER
    CANCEL_REQUESTED = False  # reset at start of each invocation

    # Split dynamic_values into nsfw category toggles and workflow edits
    num_nsfw_cats = len([k for k in CATEGORY_DISPLAY_MAPPINGS.keys() if k != "NOT_DETECTED"])
    nsfw_categories = dynamic_values[:num_nsfw_cats]
    wf_edit_values = dynamic_values[num_nsfw_cats:]

    wf1_edit_vals = wf_edit_values[: len(WF1_MAPPING)]
    wf2_edit_vals = wf_edit_values[len(WF1_MAPPING) : len(WF1_MAPPING)+len(WF2_MAPPING)]
    remaining_vals = wf_edit_values[len(WF1_MAPPING)+len(WF2_MAPPING):]

    # Extract LoRA override values --------------------------------------------------
    # New layout (flags removed):
    #  [0:6]   -> 6 LoRA paths for node 244
    #  [6:12]  -> 6 strength values for node 244
    #  [12:18] -> 6 paths for node 307
    #  [18:24] -> 6 strengths for node 307

    override_loras_244_flag = False
    override_loras_307_flag = False

    lora244_paths: list[str] = [""] * 6
    lora244_strengths: list[float] = [0.7, 0.3, 0.3, 0.3, 0.3, 0.3]

    lora307_paths: list[str] = [""] * 6
    lora307_strengths: list[float] = [0.7, 0.3, 0.3, 0.3, 0.3, 0.3]

    if remaining_vals and len(remaining_vals) >= 24:
        idx = 0
        lora244_paths = [str(v).strip() for v in remaining_vals[idx:idx+6]]; idx += 6
        lora244_strengths = [float(v) for v in remaining_vals[idx:idx+6]]; idx += 6

        lora307_paths = [str(v).strip() for v in remaining_vals[idx:idx+6]]; idx += 6
        lora307_strengths = [float(v) for v in remaining_vals[idx:idx+6]]

        override_loras_244_flag = any(p for p in lora244_paths)
        override_loras_307_flag = any(p for p in lora307_paths)

    # initialise log list early so we can safely append warnings before main loop
    log_lines: List[str] = []

    # ----------------------------------------------------------------------
    # If the user enabled direct prompt loading, create a temporary prompts
    # file and inject its path into the workflow (node class
    # "Load Prompt From File - EQX", expected id 190).
    # ----------------------------------------------------------------------

    tmp_prompt_file: str | None = None
    if load_prompts_directly:
        prompt_list_clean = (prompt_list_str or "").strip()
        if not prompt_list_clean:
            # Nothing to load – fail fast with message
            return None, {}, "Prompt list is empty while the direct prompt option is enabled.", None, "In Queue: 0", "Error", _metrics_str(), None

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix="_prompts.txt")
            os.close(tmp_fd)
            Path(tmp_path).write_text(prompt_list_clean)
            tmp_prompt_file = tmp_path
        except Exception as e:
            return None, {}, f"Could not create temporary prompt file: {e}", None, "In Queue: 0", "Error", _metrics_str(), None

    # ----------------------------------------------------------------------
    # Build modified workflow copies according to edits (only once at start
    # of batch). They will be reused across attempts, but seed will still be
    # incremented later as before.
    # ----------------------------------------------------------------------

    wf1_path_mod = _apply_edits_to_workflow(oc.WORKFLOW1_JSON, WF1_MAPPING, wf1_edit_vals)
    wf2_path_mod = _apply_edits_to_workflow(oc.WORKFLOW2_JSON, WF2_MAPPING, wf2_edit_vals)

    # --------------------------------------------------
    # Apply LoRA overrides if the user enabled either panel
    # --------------------------------------------------
    if override_loras_244_flag or override_loras_307_flag:
        try:
            data_wf2 = json.loads(Path(wf2_path_mod).read_text())

            # Helper to patch a single node
            def _patch_node(node_id: str, paths: list[str], strengths: list[float]):
                node = data_wf2.get(node_id)
                if not (node and isinstance(node, dict)):
                    return
                for i in range(1, 7):
                    path_val = paths[i - 1]
                    strength_val = strengths[i - 1]
                    if path_val:  # Only override if user provided a path
                        node.setdefault("inputs", {})[f"lora_{i}"] = {
                            "on": True,
                            "lora": path_val,
                            "strength": strength_val,
                        }

            if override_loras_244_flag:
                _patch_node("244", lora244_paths, lora244_strengths)

            if override_loras_307_flag:
                _patch_node("307", lora307_paths, lora307_strengths)

            Path(wf2_path_mod).write_text(json.dumps(data_wf2, indent=2))
        except Exception as e:
            log_lines.append(f"[WARN] Could not apply LoRA overrides: {e}")

    # When using direct prompt loading, patch node 190's file_path to our temp file
    if load_prompts_directly and tmp_prompt_file:
        def _set_prompt_file(path_json: str, file_path: str, node_class: str = "Load Prompt From File - EQX") -> bool:
            try:
                data_local = json.loads(Path(path_json).read_text())
                for node in data_local.values():
                    if isinstance(node, dict) and node.get("class_type") == node_class:
                        node.setdefault("inputs", {})["file_path"] = str(file_path)
                        Path(path_json).write_text(json.dumps(data_local, indent=2))
                        return True
            except Exception:
                pass
            return False

        _set_prompt_file(wf1_path_mod, tmp_prompt_file)

    if characteristics_text and characteristics_text.strip():
        try:
            data_local = json.loads(Path(wf1_path_mod).read_text())
            if "171" in data_local:
                data_local["171"].setdefault("inputs", {})["text"] = characteristics_text.strip()
                Path(wf1_path_mod).write_text(json.dumps(data_local, indent=2))
        except Exception as e:
            log_lines.append(f"[WARN] Could not set characteristics text for node 171: {e}")

    # ----------------------------------------------------------------------
    # DEBUG: Save final workflow JSONs sent to ComfyUI ----------------------
    # ----------------------------------------------------------------------
    try:
        debug_dir = Path(__file__).resolve().parent / "WF_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(wf1_path_mod, debug_dir / f"WF1_{ts}.json")
        shutil.copy(wf2_path_mod, debug_dir / f"WF2_{ts}.json")
    except Exception as e:
        log_lines.append(f"[WARN] Could not save debug workflow JSONs: {e}")

    success_count = 0
    attempt = 0

    # Determine the target number of successful images. If "endless" mode is
    # enabled we will iterate until the user presses **Cancel** from the UI.
    if endless_until_cancel:
        target_successes = math.inf
        # keep a sane default for batch_runs to avoid type issues later on
        batch_runs = max(1, int(batch_runs)) if batch_runs is not None else 1
    else:
        batch_runs = max(1, int(batch_runs))
        target_successes = batch_runs

    # 1. Determine selected checks --------------------------------------------------
    checks: List[str] = []
    if do_nsfw:
        checks.append("nsfw")
    if do_face_count:
        checks.append("face_count")
    if do_partial_face:
        checks.append("partial_face")

    if not checks:
        return None, {}, "You must select at least one check.", None, "In Queue: 0", "Error", _metrics_str(), None

    # Prepare output directories --------------------------------------------
    output_dir = Path(output_dir_str) if output_dir_str else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = Path(final_dir_str) if final_dir_str else Path.home() / "ComfyUI" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Album of generated images that pass checks
    album_images: List[Image.Image] = []

    while True:
        # Stop when the requested amount is reached (unless endless mode)
        if not endless_until_cancel and success_count >= target_successes:
            break

        # Check if user requested cancellation from the UI
        if CANCEL_REQUESTED:
            log_lines.append("[CANCEL] Execution cancelled by user.")
            status_str = "Cancelled"
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, None, album_images, *_boxes(pending)
            break

        attempt += 1
        # Separador visual entre intentos
        if attempt > 1:
            log_lines.append("")  # línea en blanco
        total_target = "∞" if endless_until_cancel else target_successes
        log_lines.append(
            f"========== ATTEMPT {attempt} | Passed {success_count}/{total_target} =========="
        )

        # ------------------------------------------------------------------
        # Prepare seed according to the selected mode
        # ------------------------------------------------------------------
        def _set_seed(path: str, value: int, node_class: str = "Load Prompt From File - EQX") -> bool:
            try:
                data = json.loads(Path(path).read_text())
                for node in data.values():
                    if isinstance(node, dict) and node.get("class_type") == node_class:
                        node.setdefault("inputs", {})["seed"] = int(value)
                        Path(path).write_text(json.dumps(data, indent=2))
                        return True
            except Exception:
                pass
            return False

        current_seed_val: int | None = None
        if seed_mode.lower().startswith("incre"):
            # Initialise counter only the first time we use it.
            if GLOBAL_SEED_COUNTER is None:
                GLOBAL_SEED_COUNTER = int(seed_counter_input) % (MAX_SEED_VALUE + 1)
            current_seed_val = GLOBAL_SEED_COUNTER
            GLOBAL_SEED_COUNTER = (GLOBAL_SEED_COUNTER + 1) % (MAX_SEED_VALUE + 1)
        elif seed_mode.lower().startswith("rand"):
            current_seed_val = random.randint(0, MAX_SEED_VALUE)
        else:  # Fallback to the value typed by the user (static)
            current_seed_val = min(int(seed_counter_input), MAX_SEED_VALUE)

        if _set_seed(wf1_path_mod, current_seed_val):
            log_lines.append(f"[SEED] Using seed {current_seed_val} (mode: {seed_mode}) for prompt loader")
        else:
            log_lines.append("[SEED] Warning: could not locate prompt seed node (Load Prompt From File - EQX) to update")

        # NEW: randomise the seed used by the KSampler via the 'Seed Everywhere' node (id 189)
        image_seed_val = random.randint(0, MAX_SEED_VALUE)
        if _set_seed(wf1_path_mod, image_seed_val, node_class="Seed Everywhere"):
            log_lines.append(f"[SEED] Using random image seed {image_seed_val} for KSampler (Seed Everywhere)")
        else:
            log_lines.append("[SEED] Warning: could not locate 'Seed Everywhere' node to update")

        # ------------------------------------------------------------------
        # Run Workflow 1
        # ------------------------------------------------------------------
        log_lines.append("[WF1] Running Workflow1…")
        status_str = "Generating (WF1)"
        start_time = time.time()
        try:
            oc.run_workflow(wf1_path_mod)
        except Exception as e:
            err_msg = f"Error while executing Workflow1: {e}"
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, None, album_images, *_boxes(pending)
            continue

        wf1_img_path = oc.newest_file(output_dir, after=start_time)
        if wf1_img_path is None:
            err_msg = "Could not find the resulting image from Workflow1."
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, None, album_images, *_boxes(pending)
            continue

        wf1_img_pil = _load_img(wf1_img_path)

        log_lines.append(f"[WF1] Image generated: {wf1_img_path}")

        # Show WF1 image immediately
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)

        # ----------------------------------------------------------------------
        # Build optional parameters for run_checks according to UI
        # ----------------------------------------------------------------------
        allowed_categories = None
        if do_nsfw and nsfw_categories:
            cat_keys = [k for k in sorted(CATEGORY_DISPLAY_MAPPINGS.keys()) if k != "NOT_DETECTED"]
            allowed_categories = {cat_keys[i] for i, checked in enumerate(nsfw_categories) if checked}

        log_lines.append("[CHECK] Running external checks…")
        status_str = "Checks"
        # Yield before running checks so UI reflects status change
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
        try:
            results = oc.run_checks(
                wf1_img_path,
                checks,
                allowed_categories=allowed_categories,
                confidence=confidence,
                margin=margin,
            )
        except Exception as e:
            err_msg = f"Error while executing checks: {e}"
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
            continue

        log_lines.append(f"[CHECK] Results: {json.dumps(results, indent=2, ensure_ascii=False)}")

        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)

        # -----------------------------
        failed = False
        if results.get("is_nsfw"):
            failed = True
            log_lines.append("[CHECK] Image flagged as NSFW. Pipeline stops.")

        if stop_multi_faces and results.get("face_count", 0) > 1:
            failed = True
            log_lines.append("[CHECK] More than one face detected and stop option is active.")

        if stop_partial_face and results.get("is_partial_face"):
            failed = True
            log_lines.append("[CHECK] Partial face detected and stop option is active.")

        if failed:
            # Delete the generated file to save disk space
            try:
                Path(wf1_img_path).unlink(missing_ok=True)
            except Exception as e:
                log_lines.append(f"[CLEANUP] Could not delete failed image: {e}")

            # La ejecución se considera fallida; no incrementamos success_count
            global GLOBAL_REJECTED_COUNT
            GLOBAL_REJECTED_COUNT += 1
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
            continue

        # ----------------------------------------------------------------------
        # Run Workflow 2
        # ----------------------------------------------------------------------
        verified_path = output_dir / "verified_gui.png"
        try:
            Path(wf1_img_path).replace(verified_path)
        except Exception:
            shutil.copy(wf1_img_path, verified_path)

        log_lines.append("[WF2] Running Workflow2…")
        status_str = "Generating (WF2)"
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
        start_time2 = time.time()
        overrides = {oc.LOAD_NODE_ID_WF2: {"image": str(verified_path)}}
        try:
            oc.run_workflow(wf2_path_mod, overrides=overrides)
        except Exception as e:
            err_msg = f"Error while executing Workflow2: {e}"
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
            continue

        wf2_img_path = oc.newest_file(output_dir, after=start_time2)
        if wf2_img_path is None:
            err_msg = "Could not find the resulting image from Workflow2."
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, *_boxes(pending)
            continue

        wf2_img_pil = _load_img(wf2_img_path)
        log_lines.append(f"[WF2] Image generated: {wf2_img_path}")

        # Add the approved final image to the album and display
        album_images.append(wf2_img_pil)

        # NEW: keep gallery size bounded to the last MAX_GALLERY_IMAGES
        if len(album_images) > MAX_GALLERY_IMAGES:
            album_images = album_images[-MAX_GALLERY_IMAGES:]

        status_str = "Completed"

        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, wf2_img_pil, album_images, *_boxes(pending)

        # Extract base filename from node 175 (ShowText) for this attempt
        def _extract_filename_base(path_json: str) -> str | None:
            try:
                data_json = json.loads(Path(path_json).read_text())
                val = data_json.get("175", {}).get("inputs", {}).get("text_0")
                if isinstance(val, str) and val.strip():
                    # keep only safe filename chars
                    import re
                    safe = re.sub(r"[^A-Za-z0-9._-]", "_", val.strip())
                    return safe[:255]
            except Exception:
                pass
            return None

        default_filename_base = _extract_filename_base(wf1_path_mod)

        try:
            if override_save_name and save_name_base.strip():
                base = Path(save_name_base.strip()).stem  # sanitize
                # find next available index
                idx = 1
                while (final_dir / f"{base}_{idx}.png").exists():
                    idx += 1
                final_path = final_dir / f"{base}_{idx}.png"
            elif default_filename_base:
                base = default_filename_base
                idx = 1
                while (final_dir / f"{base}_{idx}.png").exists():
                    idx += 1
                final_path = final_dir / f"{base}_{idx}.png"
            else:
                final_path = final_dir / wf2_img_path.name
            shutil.copy(wf2_img_path, final_path)
        except Exception as e:
            log_lines.append(f"Warning: could not copy final image to {final_path}: {e}")

        # Si llegamos aquí la imagen pasó todos los filtros ⇒ contamos como éxito
        success_count += 1
        global GLOBAL_APPROVED_COUNT
        GLOBAL_APPROVED_COUNT += 1

        # end loop iteration

    # end of while loop -----------------------------------------------------
    # Send final update so indicators show the correct counts (pending=0)
    sound_placeholder = _get_notification_sound() if play_sound else None

    final_pending = 0
    final_log = "\n\n".join(log_lines)
    yield None, final_log, None, album_images, *_boxes(final_pending)

    # (global mappings are updated during GUI construction; do not reassign
    # them here to avoid Python's local-variable shadowing.)

    return


# -----------------------------------------------------------------------------
# Build Gradio Interface
# -----------------------------------------------------------------------------

def launch_gui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown(
            """
            # ComfyUI Pipeline Manager + External Checks
            1. Runs Workflow 1 and shows the generated image.  
            2. Applies the selected external checks.  
            3. If all checks pass, runs Workflow 2 and shows the final image.
            """
        )

        # -------------------- Preset section at top ----------------------
        with gr.Accordion("Configuration Presets", open=False):
            with gr.Row():
                preset_name_txt = gr.Textbox(label="Preset name")
                save_preset_btn = gr.Button("Save Preset")
            with gr.Row():
                presets_dd = gr.Dropdown(label="Available presets", choices=[p.stem for p in PRESET_DIR.glob("*.json")])
                load_preset_btn = gr.Button("Load Preset")
                refresh_presets_btn = gr.Button("Refresh List")
                reset_btn = gr.Button("Reset to defaults")

        # ----------------------------------------------------------------

        run_btn = gr.Button("Run Pipeline", variant="primary")
        cancel_btn = gr.Button("Cancel", variant="stop")
        with gr.Row():
            pending_box = gr.Markdown("Pending: 0", elem_id="pending-box", elem_classes=["pending-box"])
            status_box = gr.Markdown("Idle", elem_id="status-box", elem_classes=["pending-box"])
            metrics_box = gr.Markdown("Approved: 0 | Rejected: 0", elem_id="metrics-box", elem_classes=["pending-box"])

        with gr.Row():
            with gr.Column(scale=1):
                output_dir_input = gr.Textbox(label="ComfyUI Output Folder", value=str(DEFAULT_OUTPUT_DIR))
                final_dir_input = gr.Textbox(label="Destination Folder for Final Images", value=str((Path.home()/"ComfyUI"/"final").expanduser()))

                with gr.Accordion("Checks to Perform", open=False):
                    do_nsfw_cb = gr.Checkbox(label="Detect NSFW", value=True)

                    with gr.Accordion("Select NSFW Categories to Detect", open=False) as nsfw_cat_group:
                        nsfw_cat_checkboxes = []
                        for internal, display in sorted(CATEGORY_DISPLAY_MAPPINGS.items()):
                            if internal == "NOT_DETECTED":
                                continue
                            nsfw_cat_checkboxes.append(
                                gr.Checkbox(label=display, value=(internal in DEFAULT_NSFW_CATEGORIES))
                            )

                    # Hide/show category selection depending on NSFW toggle
                    do_nsfw_cb.change(lambda x: gr.update(visible=x), do_nsfw_cb, nsfw_cat_group)

                    do_face_count_cb = gr.Checkbox(label="Count Faces", value=True)
                    do_partial_face_cb = gr.Checkbox(label="Detect Partial Face", value=True)

                    # Opciones avanzadas de detención
                    with gr.Accordion("Criteria to STOP the pipeline", open=False):
                        stop_multi_faces_cb = gr.Checkbox(
                            label="Stop if more than one face is detected", value=False
                        )
                        stop_partial_face_cb = gr.Checkbox(
                            label="Stop if face is partially visible", value=False
                        )

                    confidence_slider = gr.Slider(
                        0.1,
                        1.0,
                        0.8,
                        step=0.05,
                        label="Face detection confidence",
                        info="Higher requires stronger face match to pass (0.1-1.0)"
                    )

                    margin_slider = gr.Slider(
                        0.0,
                        2.0,
                        0.5,
                        step=0.1,
                        label="Partial face margin",
                        info="How much extra margin around face is allowed before considered partial"
                    )

                    # End of Checks accordion

                # --------------------------------------------------
                # Execution settings (batch + seed) – outside Checks accordion
                # --------------------------------------------------
                with gr.Accordion("Execution Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            batch_runs_input = gr.Number(
                                value=1,
                                minimum=1,
                                precision=0,
                                label="Batch Runs",
                                info="Number of successful final images to generate when not in endless mode"
                            )

                            endless_until_cancel_cb = gr.Checkbox(
                                label="Endless until cancel", value=False
                            )

                            play_sound_cb = gr.Checkbox(
                                label="Play sound when batch completes", value=True,
                            )

                        with gr.Column():
                            seed_mode_radio = gr.Radio(
                                choices=["Incremental", "Random", "Static"],
                                value="Incremental",
                                label="Seed mode",
                            )
                            seed_counter_input = gr.Number(
                                value=0,
                                minimum=0,
                                maximum=MAX_SEED_VALUE,
                                precision=0,
                                label="Initial/Current Seed (<= 10,000,000)",
                                info="Starting seed for Incremental mode or static seed if using Static mode"
                            )

                    # NEW: Settings Override ---------------------------------------------------
                    # Settings Override moved to a dedicated full-width section below.
                    # Variables declared here so they exist before later usage.
                    override_components: list = []
                    WF1_OVERRIDE_MAPPING: list[tuple[str, str, type]] = []
                    override_components2: list = []
                    WF2_OVERRIDE_MAPPING: list[tuple[str, str, type]] = []

                    # ----------------------------------------------------------------
                    # Additional execution overrides (kept inside Execution Settings)
                    # ----------------------------------------------------------------

                    # Custom filename ---------------------------------------------------
                    with gr.Accordion("Custom final filename", open=False):
                        override_filename_cb = gr.Checkbox(label="Override final filename", value=False)
                        filename_text = gr.Textbox(label="Base filename", value="final_image")

                    # Node 171 – Characteristics text override ---------------------------
                    characteristics_text = gr.Textbox(
                        label="Replace this with the characteristics of each girl",
                        lines=3,
                        value="A photo of GeegeeDwa1 posing next to a white chair. She has long darkbrown hair styled in pigtails and very pale skin. She wears a vibrant sexy outfit. Her expression is a smirk. The background shows a modern apartment that exudes a candid atmosphere.",
                    )

                    # Direct Prompt Loader ----------------------------------------------
                    with gr.Accordion("Load Prompt list Manually", open=False):
                        load_prompts_cb = gr.Checkbox(label="Enable direct prompt list", value=False)
                        prompts_textbox = gr.Textbox(
                            label="Prompt list (format: {{{id}}}{{positive}}{negative}, …)",
                            lines=4,
                            placeholder="{{{title}}}{{positive}}{negative}, …",
                            visible=False,
                        )

                        # Toggle visibility of the prompt textbox
                        load_prompts_cb.change(lambda x: gr.update(visible=x), inputs=[load_prompts_cb], outputs=[prompts_textbox])

                pass  # directory inputs finished

                # Log textbox (below all accordions)
                log_text = gr.Textbox(label="Log", lines=18, interactive=False)

            with gr.Column(scale=3):
                # First row: both workflow images side by side
                with gr.Row(equal_height=True):
                    wf1_img_out = gr.Image(
                        label="Workflow 1 Image",
                        interactive=False,
                        height=400,
                        elem_classes=["preview-img"],
                    )
                    wf2_img_out = gr.Image(
                        label="Workflow 2 Image",
                        interactive=False,
                        height=400,
                        elem_classes=["preview-img"],
                    )

                # Second row: gallery only
                with gr.Row():
                    try:
                        from gradio.components import Carousel  # Gradio >=4
                        album_gallery = Carousel(
                            label="Results (last 20)",
                            visible=True,
                            scale=2,
                        )
                    except ImportError:
                        album_gallery = gr.Gallery(
                            label="Results (last 20)",
                            columns=[4],
                            preview=False,
                            elem_id="results-gallery",
                            scale=2,
                        )

                    sound_audio = gr.Audio(label="Sound", autoplay=True, visible=False)

        # Dynamic workflow editors ------------------------------------------------

        # Workflow editors removed; keep mapping lists for override only
        global WF1_MAPPING, WF2_MAPPING
        wf1_components = []
        wf2_components = []
        WF1_MAPPING = WF1_OVERRIDE_MAPPING
        WF2_MAPPING = globals().get("WF2_OVERRIDE_MAPPING", [])

        # ---------------- Settings Override (Full Width) ----------------
        with gr.Accordion("Settings Override", open=False):
            override_components = []
            override_components2 = []
            WF1_OVERRIDE_MAPPING = []
            WF2_OVERRIDE_MAPPING = []

            # Helpers -------------------------------------------------
            try:
                wf1_data = json.loads(Path(oc.WORKFLOW1_JSON).read_text())
            except Exception:
                wf1_data = {}

            try:
                wf2_data = json.loads(Path(oc.WORKFLOW2_JSON).read_text())
            except Exception:
                wf2_data = {}

            def _def_val(node: dict | None, key: str, fallback: Any = None):
                if node and isinstance(node, dict):
                    return node.get("inputs", {}).get(key, fallback)
                return fallback

            def _num(label: str, value: int | float | None = None):
                return gr.Number(label=label, value=value if value is not None else 0, precision=0)

            # Layout: two columns side by side -----------------------
            with gr.Row():
                # ---------------- LEFT: Workflow 1 -----------------
                with gr.Column(scale=1):
                    gr.Markdown("### Workflow 1 Overrides")

                    # 168 – Steps KSampler 1
                    n168 = wf1_data.get("168", {})
                    comp_168 = _num("Steps KSampler 1 - 168", _def_val(n168, "steps", 20))
                    override_components.append(comp_168)
                    WF1_OVERRIDE_MAPPING.append(("168", "steps", int))

                    # 169 – Steps KSampler 2
                    n169 = wf1_data.get("169", {})
                    comp_169 = _num("Steps KSampler 2 - 169", _def_val(n169, "steps", 20))
                    override_components.append(comp_169)
                    WF1_OVERRIDE_MAPPING.append(("169", "steps", int))

                    # 170 – Prefix Title Text
                    n170 = wf1_data.get("170", {})
                    comp_170 = gr.Textbox(label="Prefix Title Text - 170", value=_def_val(n170, "text", ""))
                    override_components.append(comp_170)
                    WF1_OVERRIDE_MAPPING.append(("170", "text", str))

                    # 173 – Checkpoint
                    n173 = wf1_data.get("173", {})
                    comp_173 = gr.Textbox(label="Checkpoint - 173", value=_def_val(n173, "ckpt_name", ""))
                    override_components.append(comp_173)
                    WF1_OVERRIDE_MAPPING.append(("173", "ckpt_name", str))

                    # 176 – Width & Height
                    gr.Markdown("**Width & Height - 176**")
                    with gr.Row():
                        comp_176_w = _num("Width", _def_val(n176 := wf1_data.get("176", {}), "width", 512))
                        comp_176_h = _num("Height", _def_val(n176, "height", 512))
                    override_components.extend([comp_176_w, comp_176_h])
                    WF1_OVERRIDE_MAPPING.extend([
                        ("176", "width", int),
                        ("176", "height", int),
                    ])

                    # 180 – Text 180
                    n180 = wf1_data.get("180", {})
                    comp_180 = gr.Textbox(label="Text - 180", value=_def_val(n180, "text", ""))
                    override_components.append(comp_180)
                    WF1_OVERRIDE_MAPPING.append(("180", "text", str))

                    # 182 – Steps KSampler 3
                    n182 = wf1_data.get("182", {})
                    comp_182 = _num("Steps KSampler 3 - 182", _def_val(n182, "steps", 20))
                    override_components.append(comp_182)
                    WF1_OVERRIDE_MAPPING.append(("182", "steps", int))

                # ---------------- RIGHT: Workflow 2 ----------------
                with gr.Column(scale=1):
                    gr.Markdown("### Workflow 2 Overrides")

                    def _num2(label: str, value: int | float | None = None):
                        return gr.Number(label=label, value=value if value is not None else 0)

                    # 248 – Text 248
                    n248 = wf2_data.get("248", {})
                    comp_248 = gr.Textbox(label="Text – 248", value=_def_val(n248, "text", ""))
                    override_components2.append(comp_248)
                    WF2_OVERRIDE_MAPPING.append(("248", "text", str))

                    # 224 – Guidance
                    n224 = wf2_data.get("224", {})
                    comp_224 = _num2("Guidance", _def_val(n224, "guidance", 7.0))
                    override_components2.append(comp_224)
                    WF2_OVERRIDE_MAPPING.append(("224", "guidance", float))

                    # 226 – Grain power
                    n226 = wf2_data.get("226", {})
                    comp_226 = _num2("Grain power", _def_val(n226, "grain_power", 1.0))
                    override_components2.append(comp_226)
                    WF2_OVERRIDE_MAPPING.append(("226", "grain_power", float))

                    # 230 – Max Size
                    n230 = wf2_data.get("230", {})
                    comp_230 = _num2("Max Size", _def_val(n230, "size", 1024))
                    override_components2.append(comp_230)
                    WF2_OVERRIDE_MAPPING.append(("230", "size", int))

                    # 239 – Steps & Denoise
                    n239 = wf2_data.get("239", {})
                    comp_239_steps = _num2("Steps – 239", _def_val(n239, "steps", 20))
                    comp_239_denoise = _num2("Denoise – 239", _def_val(n239, "denoise", 0.2))
                    override_components2.extend([comp_239_steps, comp_239_denoise])
                    WF2_OVERRIDE_MAPPING.extend([
                        ("239", "steps", int),
                        ("239", "denoise", float),
                    ])

                    # 242 – CR Prompt Test 242
                    n242 = wf2_data.get("242", {})
                    comp_242 = gr.Textbox(label="CR Prompt Test – 242", value=_def_val(n242, "prompt", ""))
                    override_components2.append(comp_242)
                    WF2_OVERRIDE_MAPPING.append(("242", "prompt", str))

                    # 243 – CR Prompt Test 243
                    n243 = wf2_data.get("243", {})
                    comp_243 = gr.Textbox(label="CR Prompt Test – 243", value=_def_val(n243, "prompt", ""))
                    override_components2.append(comp_243)
                    WF2_OVERRIDE_MAPPING.append(("243", "prompt", str))

                    # 287 – Guidance 287
                    n287 = wf2_data.get("287", {})
                    comp_287 = _num2("Guidance – 287", _def_val(n287, "guidance", 7.0))
                    override_components2.append(comp_287)
                    WF2_OVERRIDE_MAPPING.append(("287", "guidance", float))

                    # 289 – Grain power 289
                    n289 = wf2_data.get("289", {})
                    comp_289 = _num2("Grain power – 289", _def_val(n289, "grain_power", 1.0))
                    override_components2.append(comp_289)
                    WF2_OVERRIDE_MAPPING.append(("289", "grain_power", float))

                    # 293 – Max size 293
                    n293 = wf2_data.get("293", {})
                    comp_293 = _num2("Max size - 293", _def_val(n293, "size", 1024))
                    override_components2.append(comp_293)
                    WF2_OVERRIDE_MAPPING.append(("293", "size", int))

                    # 302 – Steps & Denoise 302
                    n302 = wf2_data.get("302", {})
                    comp_302_steps = _num2("Steps – 302", _def_val(n302, "steps", 20))
                    comp_302_denoise = _num2("Denoise – 302", _def_val(n302, "denoise", 0.2))
                    override_components2.extend([comp_302_steps, comp_302_denoise])
                    WF2_OVERRIDE_MAPPING.extend([
                        ("302", "steps", int),
                        ("302", "denoise", float),
                    ])

                    # 305 – Prompt 305
                    n305 = wf2_data.get("305", {})
                    comp_305 = gr.Textbox(label="Prompt – 305", value=_def_val(n305, "prompt", ""))
                    override_components2.append(comp_305)
                    WF2_OVERRIDE_MAPPING.append(("305", "prompt", str))

                    # 306 – Prompt 306
                    n306 = wf2_data.get("306", {})
                    comp_306 = gr.Textbox(label="Prompt – 306", value=_def_val(n306, "prompt", ""))
                    override_components2.append(comp_306)
                    WF2_OVERRIDE_MAPPING.append(("306", "prompt", str))

                    # 311 – Text 311
                    n311 = wf2_data.get("311", {})
                    comp_311 = gr.Textbox(label="Text – 311", value=_def_val(n311, "text", ""))
                    override_components2.append(comp_311)
                    WF2_OVERRIDE_MAPPING.append(("311", "text", str))

            # Update globals and mappings -------------------------------
            WF1_MAPPING = WF1_OVERRIDE_MAPPING
            WF2_MAPPING = WF2_OVERRIDE_MAPPING

            globals().update({
                "override_components": override_components,
                "override_components2": override_components2,
                "WF1_OVERRIDE_MAPPING": WF1_OVERRIDE_MAPPING,
                "WF2_OVERRIDE_MAPPING": WF2_OVERRIDE_MAPPING,
            })

        # ---------------- LoRA OVERRIDES ----------------
        with gr.Accordion("LoRA Overrides (nodes 244 & 307)", open=False):
            with gr.Row():
                # Node 244 --------------------------------------------------
                with gr.Column(scale=1):
                    gr.Markdown("### Node 244 – LoRA list")
                    lora244_inputs = []
                    lora244_strengths = []
                    for idx in range(1, 7):
                        with gr.Row():
                            txt = gr.Textbox(label=f"lora_{idx} path", placeholder="KimberlyMc1.safetensors")
                            default_strength = 0.7 if idx == 1 else 0.3
                            sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="strength")
                        lora244_inputs.append(txt)
                        lora244_strengths.append(sl)

                # Node 307 --------------------------------------------------
                with gr.Column(scale=1):
                    gr.Markdown("### Node 307 – LoRA list")
                    lora307_inputs = []
                    lora307_strengths = []
                    for idx in range(1, 7):
                        with gr.Row():
                            txt = gr.Textbox(label=f"lora_{idx} path", placeholder="KimberlyMc1.safetensors")
                            default_strength = 0.7 if idx == 1 else 0.3
                            sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="strength")
                        lora307_inputs.append(txt)
                        lora307_strengths.append(sl)

        # -------------------------------------------------------------------
        # Dynamic workflow editors ------------------------------------------------

        # Build list of all configurable components in the exact input order
        ALL_COMPONENTS = [
            do_nsfw_cb,
            do_face_count_cb,
            do_partial_face_cb,
            stop_multi_faces_cb,
            stop_partial_face_cb,
            confidence_slider,
            margin_slider,
            output_dir_input,
            final_dir_input,
            batch_runs_input,
            endless_until_cancel_cb,
            play_sound_cb,
            seed_mode_radio,
            seed_counter_input,
            override_filename_cb,
            filename_text,
            load_prompts_cb,
            prompts_textbox,
            characteristics_text,
            *nsfw_cat_checkboxes,
            *override_components,
            *override_components2,
            *lora244_inputs,
            *lora244_strengths,
            *lora307_inputs,
            *lora307_strengths,
        ]

        DEFAULT_VALUES = [c.value for c in ALL_COMPONENTS]

        # --------------------------------------------------
        # Preset callbacks
        # --------------------------------------------------
        def _save_preset(*vals):
            *cfg_vals, preset_name = vals
            preset_name = preset_name.strip()
            if not preset_name:
                return gr.update()
            path = PRESET_DIR / f"{preset_name}.json"
            with open(path, "w") as f:
                json.dump(cfg_vals, f)
            choices = [p.stem for p in PRESET_DIR.glob("*.json")]
            return gr.update(choices=choices, value=preset_name)

        def _load_preset(selected_name):
            if not selected_name:
                return DEFAULT_VALUES
            path = PRESET_DIR / f"{selected_name}.json"
            if not path.exists():
                return DEFAULT_VALUES
            try:
                cfg_vals = json.loads(path.read_text())
                # safeguard length mismatch
                if len(cfg_vals) != len(ALL_COMPONENTS):
                    return DEFAULT_VALUES
                return cfg_vals
            except Exception:
                return DEFAULT_VALUES

        def _reset_defaults():
            return DEFAULT_VALUES

        save_preset_btn.click(_save_preset, inputs=[*ALL_COMPONENTS, preset_name_txt], outputs=[presets_dd])
        load_preset_btn.click(_load_preset, inputs=[presets_dd], outputs=ALL_COMPONENTS)
        refresh_presets_btn.click(lambda: gr.update(choices=[p.stem for p in PRESET_DIR.glob("*.json")]), outputs=[presets_dd])
        reset_btn.click(_reset_defaults, outputs=ALL_COMPONENTS)

        # -------------------- Validation -----------------------
        def _validate(*vals):
            (
                nsfw,
                face_count,
                partial_face,
                batch_runs_val,
            ) = vals
            has_check = nsfw or face_count or partial_face
            ok_batch = (batch_runs_val or 0) >= 1
            return gr.update(interactive=bool(has_check and ok_batch))

        for comp in [do_nsfw_cb, do_face_count_cb, do_partial_face_cb, batch_runs_input]:
            comp.change(
                _validate,
                inputs=[do_nsfw_cb, do_face_count_cb, do_partial_face_cb, batch_runs_input],
                outputs=[run_btn],
            )

        # Prepare inputs list in the same order expected by run_pipeline_gui
        run_btn.click(
            fn=run_pipeline_gui,
            inputs=[
                do_nsfw_cb,
                do_face_count_cb,
                do_partial_face_cb,
                stop_multi_faces_cb,
                stop_partial_face_cb,
                confidence_slider,
                margin_slider,
                output_dir_input,
                final_dir_input,
                batch_runs_input,
                endless_until_cancel_cb,
                play_sound_cb,
                seed_mode_radio,
                seed_counter_input,
                override_filename_cb,
                filename_text,
                load_prompts_cb,
                prompts_textbox,
                characteristics_text,
                *nsfw_cat_checkboxes,
                *override_components,
                *override_components2,
                *lora244_inputs,
                *lora244_strengths,
                *lora307_inputs,
                *lora307_strengths,
            ],
            outputs=[
                wf1_img_out,
                log_text,
                wf2_img_out,
                album_gallery,
                pending_box,
                status_box,
                metrics_box,
                sound_audio,
            ],
        )

        # Wire the cancel button so that it sets the global flag and updates the pending box
        cancel_btn.click(fn=_request_cancel, outputs=[pending_box, status_box, metrics_box, sound_audio])

    demo.queue().launch(server_name="0.0.0.0", server_port=18188, share=True)


if __name__ == "__main__":
    launch_gui() 