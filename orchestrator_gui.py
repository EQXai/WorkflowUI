# =============================================================================
# IMPORTS - Organized and Optimized
# =============================================================================

# Standard library imports
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
import gradio as gr
import numpy as np
from PIL import Image

# Local module imports
import orchestrator as oc
from app_config import CONFIG, CSS_STYLES
from gui_utils import (
    create_number_input, create_textbox_input, create_log_textbox,
    get_default_node_value, load_workflow_data, apply_workflow_edits,
    set_workflow_input_by_id, set_workflow_seed, load_image_safe,
    get_notification_sound, extract_filename_base, populate_lora_from_directory,
    save_preset, load_preset, list_presets, delete_preset,
    validate_pipeline_config, create_visibility_toggle_callback
)

# Pipeline utility imports
from pipeline_utils import (
    parse_dynamic_values, process_auto_lora, create_console_logger,
    setup_temporary_prompt_file, prepare_workflows, apply_lora_overrides,
    set_prompt_file, generate_seeds, setup_workflow_seeds,
    extract_prompts_normal, extract_filename_base as extract_filename_base_util,
    calculate_target_successes, should_continue_batch, build_allowed_categories,
    evaluate_check_results
)

# PromptConcatenate utility imports  
from promptconcat_utils import (
    parse_promptconcat_configs, inject_promptconcat_preview, inject_dynamic_prompts,
    get_promptconcat_debug_str, setup_characteristics_text, is_promptconcat_mode
)
from ui_sections import (
    create_preset_section, create_main_controls_section, create_status_boxes_section,
    create_directory_section, create_advanced_checks_section, create_seed_settings_section,
    create_output_prompts_section, create_images_gallery_section, create_logs_section
)
from ui_callbacks import (
    filter_presets, save_preset_wrapper, load_preset_wrapper, delete_preset_wrapper,
    rename_preset, export_preset, import_preset, update_preset_selector,
    validate_pipeline_settings, populate_lora_slots, reset_promptconcat_counters,
    generate_test_prompts, clear_test_results, apply_test_results_to_main,
    show_component_temporarily, hide_component, reset_to_defaults
)
from run_checks import CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES

# =============================================================================
# CONSTANTS - Now managed by centralized configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = Path(oc.DEFAULT_OUTPUT_DIR)

# =============================================================================
# GLOBAL STATE VARIABLES - Centralized Management
# =============================================================================

# Workflow mapping information (populated during GUI build)
WF1_MAPPING: List[tuple[str, str, type]] = []  # (node_id, input_key, original_type)
WF2_MAPPING: List[tuple[str, str, type]] = []

# Application state variables
CANCEL_REQUESTED = False
GLOBAL_SEED_COUNTER: int | None = None
GLOBAL_APPROVED_COUNT = 0
GLOBAL_REJECTED_COUNT = 0

# PromptConcatenate state (now managed by promptconcat_utils.py)
# PROMPTCONCAT_INCREMENTAL_COUNTERS and PROMPTCONCAT_DEBUG_LOG moved to promptconcat_utils.py

# Testing panel state
LAST_TEST_RESULTS: Dict[str, Any] = {}

# Notification sound function (using centralized utility)
_get_notification_sound = get_notification_sound

def _request_cancel():
    """Triggered by Cancel button: stop execution and update indicators."""
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    return (
        "Pending: 0",
        "Cancelled",
        f"Approved: {GLOBAL_APPROVED_COUNT} | Rejected: {GLOBAL_REJECTED_COUNT}",
        str(CONFIG.paths.AUDIO_FILE) if CONFIG.paths.AUDIO_FILE.exists() else None,
        "Execution cancelled by user.",
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


# Using centralized utility function
_apply_edits_to_workflow = apply_workflow_edits

# CSS styles now managed by centralized configuration

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# Using centralized utility function
_load_img = load_image_safe


def run_pipeline_gui(
    wf1_mode: str,
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
            _promptconcat_debug_str(),
        )

    status_str = "Idle"

    # Status helper ---------------------------------------------------------
    global CANCEL_REQUESTED, GLOBAL_SEED_COUNTER
    CANCEL_REQUESTED = False  # reset at start of each invocation

    # Workflow mode determination and parameter parsing using utilities
    is_promptconcat = is_promptconcat_mode(wf1_mode)

    # Parse dynamic values using centralized utility
    num_nsfw_cats = len([k for k in CATEGORY_DISPLAY_MAPPINGS.keys() if k != "NOT_DETECTED"])
    parsed_values = parse_dynamic_values(dynamic_values, num_nsfw_cats, len(WF1_MAPPING), len(WF2_MAPPING))
    
    # Extract parsed components
    nsfw_categories = parsed_values['nsfw_categories']
    wf1_edit_vals = parsed_values['wf1_edit_vals']
    wf2_edit_vals = parsed_values['wf2_edit_vals']
    auto_lora_flag = parsed_values['auto_lora_flag']
    auto_lora_dir = parsed_values['auto_lora_dir']
    lora244_paths = parsed_values['lora244_paths']
    lora244_strengths = parsed_values['lora244_strengths']
    lora307_paths = parsed_values['lora307_paths']
    lora307_strengths = parsed_values['lora307_strengths']

    # Process auto-fill LoRA using centralized utility
    lora244_paths, lora307_paths, override_loras_244_flag, override_loras_307_flag = process_auto_lora(
        auto_lora_flag, auto_lora_dir, lora244_paths, lora244_strengths, lora307_paths, lora307_strengths
    )

    # Initialize logging system using centralized utility
    log_lines: List[str] = create_console_logger()

    # Setup temporary prompt file using centralized utility
    try:
        tmp_prompt_file = setup_temporary_prompt_file(load_prompts_directly, prompt_list_str)
    except ValueError as e:
        return None, {}, "", str(e), None, "In Queue: 0", "Error", _metrics_str(), None, "No prompt list provided."
    except RuntimeError as e:
        return None, {}, "", str(e), None, "In Queue: 0", "Error", _metrics_str(), None, f"Error creating temporary file: {e}"

    # Prepare workflows using centralized utility
    wf1_path_mod, wf2_path_mod = prepare_workflows(is_promptconcat, wf1_edit_vals, wf2_edit_vals, WF1_MAPPING, WF2_MAPPING)

    # Apply LoRA overrides using centralized utility
    apply_lora_overrides(wf2_path_mod, override_loras_244_flag, override_loras_307_flag,
                        lora244_paths, lora244_strengths, lora307_paths, lora307_strengths, log_lines)

    # Setup prompt file and characteristics using centralized utilities
    if load_prompts_directly and tmp_prompt_file:
        set_prompt_file(wf1_path_mod, tmp_prompt_file)
        setup_characteristics_text(wf1_path_mod, characteristics_text, log_lines)

        # Setup PromptConcatenate configuration and preview injection
        if is_promptconcat:
            # Parse PromptConcatenate configurations
            file_configs = parse_promptconcat_configs(dynamic_values, num_nsfw_cats, len(WF1_MAPPING), len(WF2_MAPPING))
            # Inject preview prompts into workflow
            inject_promptconcat_preview(wf1_path_mod, file_configs, log_lines)

        # ----------------------------------------------------------------------
        # (The debug copy is now saved **inside** the attempt loop, once all
    #  seed values have been patched.  This guarantees that the JSON files
    #  written to WF_debug reflect exactly the parameters sent to ComfyUI.)
    # ----------------------------------------------------------------------

    success_count = 0
    attempt = 0

    # Calculate target successes and build checks configuration using utilities
    target_successes = calculate_target_successes(endless_until_cancel, batch_runs)
    if endless_until_cancel:
        batch_runs = max(1, int(batch_runs)) if batch_runs is not None else 1
    else:
        batch_runs = max(1, int(batch_runs))

    # Build checks list
    checks: List[str] = []
    if do_nsfw:
        checks.append("nsfw")
    if do_face_count:
        checks.append("face_count")
    if do_partial_face:
        checks.append("partial_face")

    if not checks:
        return None, {}, "", "You must select at least one check.", None, "In Queue: 0", "Error", _metrics_str(), None, "No checks selected."

    # Prepare output directories --------------------------------------------
    output_dir = Path(output_dir_str) if output_dir_str else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = Path(final_dir_str) if final_dir_str else Path.home() / "ComfyUI" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Album of generated images that pass checks
    album_images: List[Image.Image] = []

    # Collects human-readable seed information shown in the GUI
    seeds_log_lines: List[str] = []
    def _seeds_log_str():
        return "\n".join(seeds_log_lines)
    
    # Collects human-readable prompt information shown in the GUI
    prompt_log_lines: List[str] = []
    def _prompt_log_str():
        return "\n".join(prompt_log_lines)
    
    # Helper function to format debug log using centralized utility
    def _promptconcat_debug_str():
        return get_promptconcat_debug_str()

    while True:
        # Stop when the requested amount is reached (unless endless mode)
        if not endless_until_cancel and success_count >= target_successes:
            break

        # Ensure prompt placeholders exist for logging even if no prompt was
        # extracted/generated yet.  This prevents UnboundLocalError when the
        # pipeline runs in Normal mode (non-PromptConcatenate) because
        # *pos_prompt* and *neg_prompt* are referenced before they get a
        # value later in the loop.
        pos_prompt: str | None = None
        neg_prompt: str | None = None

        # Check if user requested cancellation from the UI
        if CANCEL_REQUESTED:
            log_lines.extend([f"[CANCEL] Execution cancelled by user."])
            status_str = "Cancelled"
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            break

        attempt += 1
        # Separador visual entre intentos
        if attempt > 1:
            log_lines.extend([""])  # línea en blanco
        total_target = "∞" if endless_until_cancel else target_successes
        log_lines.extend([
            f"========== ATTEMPT {attempt} | Passed {success_count}/{total_target} =========="
        ])
        
        # PromptConcat functionality removed

        # Setup seeds for this attempt using centralized utility
        seed_values = generate_seeds(seed_mode, seed_counter_input)
        current_seed_val = seed_values['current_seed']
        image_seed_val = seed_values['image_seed']
        ksampler2_seed_val = seed_values['ksampler2_seed']
        noise_seed_231 = seed_values['noise_seed_231']
        noise_seed_294 = seed_values['noise_seed_294']
        
        # Apply seeds to workflows using centralized utility
        setup_workflow_seeds(wf1_path_mod, wf2_path_mod, seed_values, log_lines)

        # ------------------------------------------------------------------
        # Collect seeds for logging ----------------------------------------
        # ------------------------------------------------------------------
        # Visual separation between attempts
        if attempt > 1:
            seeds_log_lines.append("")

        # Build structured seed log block with ATTEMPT header
        total_target = "∞" if endless_until_cancel else target_successes
        seed_block_gui = [
            f"========== ATTEMPT {attempt} | Passed {success_count}/{total_target} ==========",
            "WF1:",
            f"  Prompt loader seed (190): {current_seed_val}",
            f"  Seed_KSampler_1 (189): {image_seed_val}",
            f"  Seed_KSampler_2 (285): {ksampler2_seed_val}",
            "WF2:",
            f"  noise_seed (231): {noise_seed_231}",
            f"  noise_seed (294): {noise_seed_294}",
        ]
        seed_block_cli = [
            "[SEED] WF1:",
            f"[SEED]   Prompt loader seed (190): {current_seed_val}",
            f"[SEED]   Seed_KSampler_1 (189): {image_seed_val}",
            f"[SEED]   Seed_KSampler_2 (285): {ksampler2_seed_val}",
            "[SEED] WF2:",
            f"[SEED]   noise_seed (231): {noise_seed_231}",
            f"[SEED]   noise_seed (294): {noise_seed_294}",
        ]

        # Add seeds to dedicated log pane
        seeds_log_lines.extend(seed_block_gui)

        # Do not add seed details to main log to avoid duplication
        # log_lines.extend(seed_block_gui)

        # Handle prompt generation/extraction based on mode
        if is_promptconcat:
            # Parse configurations and inject dynamic prompts for PromptConcatenate
            file_configs = parse_promptconcat_configs(dynamic_values, num_nsfw_cats, len(WF1_MAPPING), len(WF2_MAPPING))
            pos_prompt, neg_prompt = inject_dynamic_prompts(wf1_path_mod, file_configs, current_seed_val, attempt)
            log_lines.extend([f"[PROMPTCONCAT] Generated prompts for attempt {attempt}"])
        else:
            # Extract prompts for Normal mode using centralized utility
            pos_prompt, neg_prompt = extract_prompts_normal(wf1_path_mod, current_seed_val)
            log_lines.extend([f"[NORMAL] Extracted prompts for attempt {attempt}"])

        # Log prompts to dedicated prompt log (works for both modes)
        if attempt > 1:
            prompt_log_lines.append("")
        
        total_target = "∞" if endless_until_cancel else target_successes
        prompt_log_lines.extend([
            f"========== ATTEMPT {attempt} | Passed {success_count}/{total_target} ==========",
            f"Positive Prompt: {pos_prompt if pos_prompt is not None else 'No prompt found'}",
            f"Negative Prompt: {neg_prompt if neg_prompt is not None else 'No prompt found'}",
        ])

        # ------------------------------------------------------------------
        # Run Workflow 1
        # ------------------------------------------------------------------
        log_lines.extend([f"[WF1] Running Workflow1…"])
        status_str = "Generating (WF1)"

        # --- Yield *before* starting the lengthy WF1 call so the GUI shows
        #     the attempt header and seeds immediately ---------------------
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield None, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)

        start_time = time.time()
        try:
            oc.run_workflow(wf1_path_mod)
        except Exception as e:
            err_msg = f"Error while executing Workflow1: {e}"
            log_lines.extend([err_msg])
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        wf1_img_path = oc.newest_file(output_dir, after=start_time)
        if wf1_img_path is None:
            err_msg = "Could not find the resulting image from Workflow1."
            log_lines.extend([err_msg])
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield None, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        wf1_img_pil = _load_img(wf1_img_path)

        log_lines.extend([f"[WF1] Image generated: {wf1_img_path}"])

        # Show WF1 image immediately
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)

        # ----------------------------------------------------------------------
        # Build optional parameters for run_checks according to UI
        # ----------------------------------------------------------------------
        allowed_categories = None
        if do_nsfw and nsfw_categories:
            cat_keys = [k for k in sorted(CATEGORY_DISPLAY_MAPPINGS.keys()) if k != "NOT_DETECTED"]
            allowed_categories = {cat_keys[i] for i, checked in enumerate(nsfw_categories) if checked}

        log_lines.extend([f"[CHECK] Running external checks…"])
        status_str = "Checks"
        # Yield before running checks so UI reflects status change
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
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
            log_lines.extend([err_msg])
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        # Pretty-print results on multiple lines for readability ------------
        log_lines.append("[CHECK] Results:")
        for _ln in json.dumps(results, indent=2, ensure_ascii=False).splitlines():
            log_lines.append(f"  {_ln}")

        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)

        # Evaluate check results using centralized utility
        failed = evaluate_check_results(results, stop_multi_faces, stop_partial_face, log_lines)

        if failed:
            # Delete the generated file to save disk space
            try:
                Path(wf1_img_path).unlink(missing_ok=True)
            except Exception as e:
                log_lines.extend([f"[CLEANUP] Could not delete failed image: {e}"])

            # La ejecución se considera fallida; no incrementamos success_count
            global GLOBAL_REJECTED_COUNT
            GLOBAL_REJECTED_COUNT += 1
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        # ----------------------------------------------------------------------
        # Run Workflow 2
        # ----------------------------------------------------------------------
        verified_path = output_dir / "verified_gui.png"
        try:
            Path(wf1_img_path).replace(verified_path)
        except Exception:
            shutil.copy(wf1_img_path, verified_path)

        # NEW: remove the original Workflow-1 image so we only keep the temporary verified copy
        try:
            Path(wf1_img_path).unlink(missing_ok=True)
        except Exception as e:
            log_lines.extend([f"[CLEANUP] Could not delete intermediate WF1 image: {e}"])

        log_lines.extend([f"[WF2] Running Workflow2…"])
        status_str = "Generating (WF2)"
        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
        start_time2 = time.time()
        overrides = {oc.LOAD_NODE_ID_WF2: {"image": str(verified_path)}}
        try:
            oc.run_workflow(wf2_path_mod, overrides=overrides)
        except Exception as e:
            err_msg = f"Error while executing Workflow2: {e}"
            log_lines.extend([err_msg])
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        wf2_img_path = oc.newest_file(output_dir, after=start_time2)
        if wf2_img_path is None:
            err_msg = "Could not find the resulting image from Workflow2."
            log_lines.extend([err_msg])
            current_log = "\n\n".join(log_lines)
            pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)
            continue

        wf2_img_pil = _load_img(wf2_img_path)
        log_lines.extend([f"[WF2] Image generated: {wf2_img_path}"])

        # Add the approved final image to the album and display
        album_images.append(wf2_img_pil)

        # Keep gallery size bounded to the configured maximum
        if len(album_images) > CONFIG.ui.MAX_GALLERY_IMAGES:
            album_images = album_images[-CONFIG.ui.MAX_GALLERY_IMAGES:]

        status_str = "Completed"

        current_log = "\n\n".join(log_lines)
        pending = "∞" if endless_until_cancel else max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(pending)

        # Extract base filename from node 175 (ShowText) for this attempt
        # Extract filename base using centralized utility
        filename_base = extract_filename_base_util(wf1_path_mod, wf2_path_mod, is_promptconcat, pos_prompt, neg_prompt, override_save_name, save_name_base)
        
        # Filename base calculation moved to pipeline_utils.py

        default_filename_base = extract_filename_base_util(wf1_path_mod, wf2_path_mod, is_promptconcat, pos_prompt, neg_prompt, override_save_name, save_name_base)

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
            log_lines.extend([f"[FINAL] Image saved as: {final_path}"])
            # NEW: remove the original Workflow-2 image written by ComfyUI to avoid duplicates
            try:
                Path(wf2_img_path).unlink(missing_ok=True)
            except Exception as e_cleanup:
                log_lines.extend([f"[CLEANUP] Could not delete intermediate WF2 image: {e_cleanup}"])

        except Exception as e:
            log_lines.extend([f"Warning: could not copy final image to {final_path}: {e}"])

        # Si llegamos aquí la imagen pasó todos los filtros ⇒ contamos como éxito
        success_count += 1
        global GLOBAL_APPROVED_COUNT
        GLOBAL_APPROVED_COUNT += 1

        # ------------------------------------------------------------------
        # DEBUG: Save workflow JSONs with final seeds for this attempt ------
        # ------------------------------------------------------------------
        try:
            debug_dir = Path(__file__).resolve().parent / "WF_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts_attempt = time.strftime("%Y%m%d_%H%M%S")
            shutil.copy(wf1_path_mod, debug_dir / f"WF1_{ts_attempt}_attempt{attempt}.json")
            shutil.copy(wf2_path_mod, debug_dir / f"WF2_{ts_attempt}_attempt{attempt}.json")
        except Exception as e:
            log_lines.append(f"[WARN] Could not save debug workflow JSONs: {e}")

        # end loop iteration

    # Send final update so indicators show the correct counts (pending=0)
    sound_placeholder = _get_notification_sound() if play_sound else None

    final_pending = 0
    final_log = "\n\n".join(log_lines)
    yield None, final_log, _seeds_log_str(), _prompt_log_str(), album_images, *_boxes(final_pending)

    # (global mappings are updated during GUI construction; do not reassign
    # them here to avoid Python's local-variable shadowing.)

    return


# -----------------------------------------------------------------------------
# Build Gradio Interface
# -----------------------------------------------------------------------------

def launch_gui():
    with gr.Blocks(css=CSS_STYLES) as demo:
        gr.Markdown("# WorkflowUI")

        # -------------------- Preset section (modularized) ----------------------
        preset_components = create_preset_section()
        (filter_txt, presets_dd, preset_name_txt, save_preset_btn,
         load_preset_btn, delete_preset_btn, export_btn, reset_btn,
         rename_txt, rename_btn, import_file) = preset_components

        # Helper functions now imported from ui_callbacks

        # --- Filter button updates both selectors --------------------
        filter_txt.change(
            lambda txt: gr.update(choices=filter_presets(txt), value=None),
            inputs=[filter_txt],
            outputs=[presets_dd],
        )

        # --- Sync list->dropdown so existing callbacks keep working ---
        presets_dd.change(lambda x: x, inputs=[presets_dd], outputs=[presets_dd])

        # --- Delete preset -------------------------------------------
        delete_preset_btn.click(
            lambda sel: update_preset_selector(None) if (delete_preset_wrapper(sel) or True) else gr.update(),
            inputs=[presets_dd],
            outputs=[presets_dd],
        )

        # --- Rename preset -------------------------------------------
        rename_btn.click(
            lambda old, new: update_preset_selector(new) if (rename_preset(old, new) or True) else gr.update(),
            inputs=[presets_dd, rename_txt],
            outputs=[presets_dd],
        )

        # --- Export preset (unchanged, uses current selection) -------
        export_btn.click(export_preset, inputs=[presets_dd], outputs=[import_file])

        # --- Import preset (returns update objects) ------------------
        import_file.upload(
            lambda f: update_preset_selector(None) if (import_preset(f) or True) else gr.update(),
            inputs=[import_file],
            outputs=[presets_dd],
        )

        # Main controls and status (modularized)
        batch_runs_input, run_btn, cancel_btn, endless_until_cancel_cb, play_sound_cb = create_main_controls_section()
        pending_box, status_box, metrics_box = create_status_boxes_section()

        with gr.Row():
            with gr.Column(scale=1):
                output_dir_input, final_dir_input = create_directory_section()

                checks_components = create_advanced_checks_section()
                (do_nsfw_cb, do_face_count_cb, do_partial_face_cb, nsfw_cat_checkboxes,
                 nsfw_cat_group, stop_multi_faces_cb, stop_partial_face_cb,
                 confidence_slider, margin_slider) = checks_components

                # Seed settings (modularized)
                seed_mode_radio, seed_counter_input = create_seed_settings_section()

                # Output & prompts (modularized)
                output_components = create_output_prompts_section()
                (override_filename_cb, filename_text, characteristics_text,
                 load_prompts_cb, prompts_textbox) = output_components

                pass  # directory inputs finished

            # Middle column: Images and Gallery (modularized)
            with gr.Column(scale=2):
                wf1_img_out, album_gallery, sound_audio = create_images_gallery_section()

            # Right column: Logs (modularized)
            with gr.Column(scale=1):
                log_text, seeds_log_text, prompt_log_text = create_logs_section()

        # Dynamic workflow editors ------------------------------------------------

        # Initialize override mappings before use
        WF1_OVERRIDE_MAPPING = []
        WF2_OVERRIDE_MAPPING = []

        # Workflow editors removed; keep mapping lists for override only
        global WF1_MAPPING, WF2_MAPPING
        wf1_components = []
        wf2_components = []
        WF1_MAPPING = WF1_OVERRIDE_MAPPING
        WF2_MAPPING = globals().get("WF2_OVERRIDE_MAPPING", [])

        # ---------------- Settings Override (Full Width) ----------------
        with gr.Accordion("🔧 Advanced Workflow Settings", open=False):
            override_components = []
            override_components2 = []

            # Helpers -------------------------------------------------
            try:
                wf1_data = json.loads(Path(oc.WORKFLOW1_JSON).read_text())
            except Exception:
                wf1_data = {}

            try:
                wf2_data = json.loads(Path(oc.WORKFLOW2_JSON).read_text())
            except Exception:
                wf2_data = {}

            # Using centralized utility functions
            _def_val = get_default_node_value

            # Layout: two columns side by side -----------------------
            with gr.Row():
                # ---------------- LEFT: Workflow 1 -----------------
                with gr.Column(scale=1):
                    # Workflow 1 Mode -------------------
                    wf1_mode_radio = gr.Radio(
                        choices=["Normal", "PromptConcatenate"],
                        value="Normal",
                        label="Workflow 1 Mode",
                    )

                    gr.Markdown("### Workflow 1")

                    # 168 – Steps KSampler 1
                    n168 = wf1_data.get("168", {})
                    comp_168 = create_number_input("Steps KSampler 1", _def_val(n168, "steps", 20))
                    override_components.append(comp_168)
                    WF1_OVERRIDE_MAPPING.append(("168", "steps", int))

                    # 169 – Steps KSampler 2
                    n169 = wf1_data.get("169", {})
                    comp_169 = create_number_input("Steps KSampler 2", _def_val(n169, "steps", 60))
                    override_components.append(comp_169)
                    WF1_OVERRIDE_MAPPING.append(("169", "steps", int))

                    # 170 – Prefix Title Text
                    n170 = wf1_data.get("170", {})
                    comp_170 = gr.Textbox(label="Prefix Title", value=_def_val(n170, "text", "GirlName"))
                    override_components.append(comp_170)
                    WF1_OVERRIDE_MAPPING.append(("170", "text", str))

                    # 173 – Checkpoint
                    n173 = wf1_data.get("173", {})
                    comp_173 = gr.Textbox(label="Checkpoint", value=_def_val(n173, "ckpt_name", ""))
                    override_components.append(comp_173)
                    WF1_OVERRIDE_MAPPING.append(("173", "ckpt_name", str))

                    # 176 – Width & Height
                    gr.Markdown("**Resolution**")
                    with gr.Row():
                        comp_176_w = create_number_input("Width", _def_val(n176 := wf1_data.get("176", {}), "width", 512))
                        comp_176_h = create_number_input("Height", _def_val(n176, "height", 512))
                    override_components.extend([comp_176_w, comp_176_h])
                    WF1_OVERRIDE_MAPPING.extend([
                        ("176", "width", int),
                        ("176", "height", int),
                    ])

                    # 180 – Face Prompt Node
                    n180 = wf1_data.get("180", {})
                    comp_180 = gr.Textbox(label="Face Prompt Node", value=_def_val(n180, "text", ""))
                    override_components.append(comp_180)
                    WF1_OVERRIDE_MAPPING.append(("180", "text", str))

                    # 182 – Steps KSampler 3
                    n182 = wf1_data.get("182", {})
                    comp_182 = create_number_input("Steps KSampler 3", _def_val(n182, "steps", 20))
                    override_components.append(comp_182)
                    WF1_OVERRIDE_MAPPING.append(("182", "steps", int))

                    # 190 – File Path (Load Prompt From File)
                    n190 = wf1_data.get("190", {})
                    _f_raw = _def_val(n190, "file_path", "stand1.txt")
                    from pathlib import Path as _Pth
                    _f_default = _Pth(_f_raw).name if isinstance(_f_raw, str) else "stand1.txt"
                    comp_190 = gr.Textbox(
                        label="File Path - 190 (relative to texts/)",
                        value=_f_default,
                        placeholder="my_prompts.txt",
                        info="Only Filename"
                    )
                    override_components.append(comp_190)
                    WF1_OVERRIDE_MAPPING.append(("190", "file_path", str))



                # ---------------- RIGHT: Workflow 2 ----------------
                with gr.Column(scale=1):
                    gr.Markdown("### Workflow 2")

                    # Using centralized utility function for numbers

                    # 248 – Suffix Final File Name - 248
                    n248 = wf2_data.get("248", {})
                    comp_248 = gr.Textbox(label="Suffix Final File Name - 248", value=_def_val(n248, "text", "SUFFIX"))
                    override_components2.append(comp_248)
                    WF2_OVERRIDE_MAPPING.append(("248", "text", str))

                    # 224 – Guidance
                    n224 = wf2_data.get("224", {})
                    comp_224 = create_number_input("Guidance", _def_val(n224, "guidance", 7.0), precision=1)
                    override_components2.append(comp_224)
                    WF2_OVERRIDE_MAPPING.append(("224", "guidance", float))

                    # 226 – Grain power
                    n226 = wf2_data.get("226", {})
                    comp_226 = create_number_input("Grain power", _def_val(n226, "grain_power", 1.0), precision=1)
                    override_components2.append(comp_226)
                    WF2_OVERRIDE_MAPPING.append(("226", "grain_power", float))

                    # 230 – Max Size
                    n230 = wf2_data.get("230", {})
                    comp_230 = create_number_input("Max Size", _def_val(n230, "size", 1024))
                    override_components2.append(comp_230)
                    WF2_OVERRIDE_MAPPING.append(("230", "size", int))

                    # 239 – Steps & Denoise
                    n239 = wf2_data.get("239", {})
                    comp_239_steps = create_number_input("Steps – 239", _def_val(n239, "steps", 20))
                    comp_239_denoise = create_number_input("Denoise – 239", _def_val(n239, "denoise", 0.75), precision=1)
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
                    comp_287 = create_number_input("Guidance – 287", _def_val(n287, "guidance", 7.0), precision=1)
                    override_components2.append(comp_287)
                    WF2_OVERRIDE_MAPPING.append(("287", "guidance", float))

                    # 289 – Grain power 289
                    n289 = wf2_data.get("289", {})
                    comp_289 = create_number_input("Grain power – 289", _def_val(n289, "grain_power", 1.0), precision=1)
                    override_components2.append(comp_289)
                    WF2_OVERRIDE_MAPPING.append(("289", "grain_power", float))

                    # 293 – Max size 293
                    n293 = wf2_data.get("293", {})
                    comp_293 = create_number_input("Max size - 293", _def_val(n293, "size", 1024))
                    override_components2.append(comp_293)
                    WF2_OVERRIDE_MAPPING.append(("293", "size", int))

                    # 302 – Steps & Denoise 302
                    n302 = wf2_data.get("302", {})
                    comp_302_steps = create_number_input("Steps – 302", _def_val(n302, "steps", 20))
                    comp_302_denoise = create_number_input("Denoise – 302", _def_val(n302, "denoise", 0.75), precision=1)
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
        with gr.Accordion("🎨 LoRA Configuration", open=False):
            gr.Markdown("*Leave blank to use workflow defaults*")
            
            with gr.Accordion("Auto-fill from directory", open=False):
                auto_lora_cb = gr.Checkbox(label="Enable auto-fill", value=False)
                lora_dir_tb = gr.Textbox(label="LoRA directory", placeholder="/path/to/lora/folder")
                populate_btn = gr.Button("Auto Select")
            with gr.Row():
                # Node 244 --------------------------------------------------
                with gr.Column(scale=1):
                    gr.Markdown("### Node 244")
                    lora244_inputs = []
                    lora244_strengths = []
                    for idx in range(1, 7):
                        with gr.Row():
                            txt = gr.Textbox(label=f"LoRA {idx}", placeholder="KimberlyMc1.safetensors")
                            default_strength = 0.7 if idx == 1 else 0.3
                            sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="str")
                        lora244_inputs.append(txt)
                        lora244_strengths.append(sl)

                # Node 307 --------------------------------------------------
                with gr.Column(scale=1):
                    gr.Markdown("### Node 307")
                    lora307_inputs = []
                    lora307_strengths = []
                    for idx in range(1, 7):
                        with gr.Row():
                            txt = gr.Textbox(label=f"LoRA {idx}", placeholder="KimberlyMc1.safetensors")
                            default_strength = 0.7 if idx == 1 else 0.3
                            sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="str")
                        lora307_inputs.append(txt)
                        lora307_strengths.append(sl)

            # LoRA population callback now imported from ui_callbacks

            # Wire button ---------------------------------------------------------
            populate_btn.click(
                populate_lora_slots,
                inputs=[lora_dir_tb],
                outputs=[*lora244_inputs, *lora307_inputs],
            )

        # ---------------- PromptConcatenate Configuration ----------------
        with gr.Accordion("🎯 PromptConcatenate Setup", open=False) as promptconcat_accordion:
            gr.Markdown("**Configure how each prompt file is selected**")
            
            # Build the file list dynamically from txt/ directory
            txt_base_dir = Path(__file__).resolve().parent / "txt"
            promptconcat_components = []
            
            if txt_base_dir.exists():
                # Organize files by directory
                positive_files = []
                negative_files = []
                
                for subdir in ["positive", "negative"]:
                    subdir_path = txt_base_dir / subdir
                    if subdir_path.exists():
                        files = sorted(subdir_path.glob("*.txt"))
                        if subdir == "positive":
                            positive_files = [f.name for f in files]
                        else:
                            negative_files = [f.name for f in files]
                
                # Two-column layout
                with gr.Row():
                    # Left column: Positive files
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Positive")
                        for filename in positive_files:
                            with gr.Group():
                                gr.Markdown(f"**{filename}**")
                                
                                mode_radio = gr.Radio(
                                    choices=["default", "fixed", "incremental", "randomized", "index"],
                                    value="default",
                                    label="Mode",
                                    info="default: seed-based | fixed: specific line | incremental: cycle | randomized: random | index: from testing"
                                )
                                
                                fixed_index = gr.Number(
                                    value=0, minimum=0, precision=0,
                                    label="Index",
                                    info="Line number (0-based)",
                                    visible=False
                                )
                                
                                # Show/hide index field based on mode selection
                                def make_toggle_fixed_index(fixed_comp):
                                    def toggle_fixed_index(mode):
                                        return gr.update(visible=(mode in ["fixed", "index"]))
                                    return toggle_fixed_index
                                
                                mode_radio.change(
                                    make_toggle_fixed_index(fixed_index),
                                    inputs=[mode_radio],
                                    outputs=[fixed_index]
                                )
                                
                                promptconcat_components.extend([mode_radio, fixed_index])
                    
                    # Right column: Negative files
                    with gr.Column(scale=1):
                        gr.Markdown("### ➖ Negative")
                        for filename in negative_files:
                            with gr.Group():
                                gr.Markdown(f"**{filename}**")
                                
                                mode_radio = gr.Radio(
                                    choices=["default", "fixed", "incremental", "randomized", "index"],
                                    value="default",
                                    label="Mode",
                                    info="default: seed-based | fixed: specific line | incremental: cycle | randomized: random | index: from testing"
                                )
                                
                                fixed_index = gr.Number(
                                    value=0, minimum=0, precision=0,
                                    label="Index",
                                    info="Line number (0-based)",
                                    visible=False
                                )
                                
                                # Show/hide index field based on mode selection
                                def make_toggle_fixed_index(fixed_comp):
                                    def toggle_fixed_index(mode):
                                        return gr.update(visible=(mode in ["fixed", "index"]))
                                    return toggle_fixed_index
                                
                                mode_radio.change(
                                    make_toggle_fixed_index(fixed_index),
                                    inputs=[mode_radio],
                                    outputs=[fixed_index]
                                )
                                
                                promptconcat_components.extend([mode_radio, fixed_index])
            
            # Reset button for incremental counters
            with gr.Row():
                reset_counters_btn = gr.Button("🔄 Reset Counters", variant="secondary")
                gr.Markdown("*Reset incremental counters to 0*")
            
            reset_status = gr.Textbox(label="Reset Status", visible=False)
            reset_counters_btn.click(
                reset_promptconcat_counters,
                outputs=[reset_status]
            )
            
            # Show reset status temporarily using imported callbacks
            reset_counters_btn.click(show_component_temporarily, outputs=[reset_status])
            reset_status.change(lambda x: hide_component() if x else gr.update(), inputs=[reset_status], outputs=[reset_status])
        
        # ---------------- Prompt Testing Section ----------------
        with gr.Accordion("🧪 Prompt Testing & Preview", open=False) as prompt_testing_accordion:
            gr.Markdown("**Test configurations and apply results to main panel**")
            
            # Testing configuration row
            with gr.Row():
                test_seed = gr.Number(
                    value=42, label="Test Seed", info="Seed for testing"
                )
                test_mode = gr.Radio(
                    choices=["deterministic", "incremental", "random"],
                    value="deterministic",
                    label="Test Mode", info="Default for all files"
                )
            
            # Build the testing components (similar to main panel)
            test_components = []
            
            if txt_base_dir.exists():
                # Two-column layout for testing
                with gr.Row():
                    # Left column: Positive files (Testing)
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Test Positive")
                        for filename in positive_files:
                            with gr.Group():
                                gr.Markdown(f"**{filename}**")
                                
                                test_mode_radio = gr.Radio(
                                    choices=["default", "fixed", "incremental", "randomized"],
                                    value="default",
                                    label="Mode",
                                    info="default: seed | fixed: specific | incremental: cycle | randomized: random"
                                )
                                
                                test_fixed_index = gr.Number(
                                    value=0, minimum=0, precision=0,
                                    label="Index",
                                    info="Line number (0-based)",
                                    visible=False
                                )
                                
                                # Show/hide fixed index based on mode selection
                                def make_test_toggle_fixed_index(fixed_comp):
                                    def toggle_fixed_index(mode):
                                        return gr.update(visible=(mode == "fixed"))
                                    return toggle_fixed_index
                                
                                test_mode_radio.change(
                                    make_test_toggle_fixed_index(test_fixed_index),
                                    inputs=[test_mode_radio],
                                    outputs=[test_fixed_index]
                                )
                                
                                test_components.extend([test_mode_radio, test_fixed_index])
                    
                    # Right column: Negative files (Testing)
                    with gr.Column(scale=1):
                        gr.Markdown("### ➖ Test Negative")
                        for filename in negative_files:
                            with gr.Group():
                                gr.Markdown(f"**{filename}**")
                                
                                test_mode_radio = gr.Radio(
                                    choices=["default", "fixed", "incremental", "randomized"],
                                    value="default",
                                    label="Mode",
                                    info="default: seed | fixed: specific | incremental: cycle | randomized: random"
                                )
                                
                                test_fixed_index = gr.Number(
                                    value=0, minimum=0, precision=0,
                                    label="Index",
                                    info="Line number (0-based)",
                                    visible=False
                                )
                                
                                # Show/hide fixed index based on mode selection
                                def make_test_toggle_fixed_index(fixed_comp):
                                    def toggle_fixed_index(mode):
                                        return gr.update(visible=(mode == "fixed"))
                                    return toggle_fixed_index
                                
                                test_mode_radio.change(
                                    make_test_toggle_fixed_index(test_fixed_index),
                                    inputs=[test_mode_radio],
                                    outputs=[test_fixed_index]
                                )
                                
                                test_components.extend([test_mode_radio, test_fixed_index])
            
            # Testing action buttons
            with gr.Row():
                test_generate_btn = gr.Button("🧪 Generate Test Prompts", variant="primary")
                apply_to_main_btn = gr.Button("⚡ Apply Test Results to Main Panel", variant="secondary")
                test_clear_btn = gr.Button("🗑️ Clear Test Results", variant="secondary")
            
            # Testing results section
            with gr.Row():
                with gr.Column(scale=1):
                    test_positive_result = gr.Textbox(
                        label="Test Positive Prompt",
                        lines=4,
                        interactive=False
                    )
                    test_negative_result = gr.Textbox(
                        label="Test Negative Prompt", 
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    test_indices_result = gr.Textbox(
                        label="Selected Indices",
                        lines=3,
                        interactive=False,
                        info="Shows which line was selected from each file"
                    )
                    test_debug_result = gr.Textbox(
                        label="Test Debug Info",
                        lines=5,
                        interactive=False
                    )
            
            # Status for apply operation
            apply_status = gr.Textbox(label="Apply Status", visible=False)

        # PromptConcatenate debug log
        promptconcat_debug_log = gr.Textbox(
            label="PromptConcatenate Debug Log",
            lines=8,
            interactive=False,
            visible=True,
            value=""
        )

        # -------------------------------------------------------------------
        # Dynamic workflow editors ------------------------------------------------

        # Build list of all configurable components in the exact input order
        ALL_COMPONENTS = [
            wf1_mode_radio,
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
            auto_lora_cb,
            lora_dir_tb,
            *lora244_inputs,
            *lora244_strengths,
            *lora307_inputs,
            *lora307_strengths,
            *promptconcat_components,
        ]

        DEFAULT_VALUES = [c.value for c in ALL_COMPONENTS]

        # --------------------------------------------------
        # Preset callbacks
        # --------------------------------------------------
        # Preset functions now imported from ui_callbacks

        # --------------------------------------------------
        # Prompt Testing callbacks
        # --------------------------------------------------
        # Test prompt callback functions now imported from ui_callbacks

        # Now that ALL_COMPONENTS is defined, wire callbacks that depend on it
        save_preset_btn.click(save_preset_wrapper, inputs=[*ALL_COMPONENTS, preset_name_txt], outputs=[presets_dd])
        load_preset_btn.click(lambda sel: load_preset_wrapper(sel, DEFAULT_VALUES), inputs=[presets_dd], outputs=ALL_COMPONENTS)
        reset_btn.click(lambda: reset_to_defaults(DEFAULT_VALUES), outputs=ALL_COMPONENTS)

        # -------------------- Validation -----------------------
        for comp in [do_nsfw_cb, do_face_count_cb, do_partial_face_cb, batch_runs_input]:
            comp.change(
                validate_pipeline_settings,
                inputs=[do_nsfw_cb, do_face_count_cb, do_partial_face_cb, batch_runs_input],
                outputs=[run_btn],
            )

        # Prepare inputs list in the same order expected by run_pipeline_gui
        run_btn.click(
            fn=run_pipeline_gui,
            inputs=[
                wf1_mode_radio,
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
                auto_lora_cb,
                lora_dir_tb,
                *lora244_inputs,
                *lora244_strengths,
                *lora307_inputs,
                *lora307_strengths,
                *promptconcat_components,
            ],
            outputs=[
                wf1_img_out,
                log_text,
                seeds_log_text,
                prompt_log_text,
                album_gallery,
                pending_box,
                status_box,
                metrics_box,
                sound_audio,
                promptconcat_debug_log,
            ],
        )

        # Wire the cancel button so that it sets the global flag and updates the pending box
        cancel_btn.click(fn=_request_cancel, outputs=[pending_box, status_box, metrics_box, sound_audio, promptconcat_debug_log])

        # --------------------------------------------------
        # Wire Prompt Testing callbacks
        # --------------------------------------------------
        test_generate_btn.click(
            fn=generate_test_prompts,
            inputs=[test_seed, test_mode, *test_components],
            outputs=[test_positive_result, test_negative_result, test_indices_result, test_debug_result]
        )
        
        test_clear_btn.click(
            fn=clear_test_results,
            outputs=[test_positive_result, test_negative_result, test_indices_result, test_debug_result]
        )
        
        apply_to_main_btn.click(
            fn=apply_test_results_to_main,
            outputs=[apply_status, *promptconcat_components]
        )
        
        # Show apply status temporarily using imported callbacks
        apply_to_main_btn.click(show_component_temporarily, outputs=[apply_status])
        apply_status.change(lambda x: hide_component() if x else gr.update(), inputs=[apply_status], outputs=[apply_status])

    demo.queue().launch(
        server_name=CONFIG.ui.SERVER_NAME,
        server_port=CONFIG.ui.SERVER_PORT,
        share=CONFIG.ui.SHARE,
        inbrowser=CONFIG.ui.INBROWSER
    )


if __name__ == "__main__":
    launch_gui() 