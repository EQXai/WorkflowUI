import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
from PIL import Image
import shutil
import tempfile

# Local modules
import orchestrator as oc  # our previously created orchestrator.py

# Constants
DEFAULT_OUTPUT_DIR = Path(oc.DEFAULT_OUTPUT_DIR)

# Extra constants for advanced check configuration
from run_checks import CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES

# -------------------------------------------------------------
# Dynamic form helpers for workflow editing
# -------------------------------------------------------------

# Will be populated during GUI build so that run_pipeline_gui can
# access the mapping information.
WF1_MAPPING: list[tuple[str, str, type]] = []  # (node_id, input_key, original_type)
WF2_MAPPING: list[tuple[str, str, type]] = []

# Flag used to signal cancellation from the GUI
CANCEL_REQUESTED = False

def _request_cancel():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    return "Cancellation requested. Finishing current step…"

def _build_workflow_editor(json_path: str):
    """Return (components, mapping) for the workflow located at *json_path*.

    components: list of Gradio components created.
    mapping: list of tuples (node_id, input_key, original_type) in the same order
             as the returned components so that their values can later be mapped
             back into the JSON structure.
    """

    from gradio.components import Textbox, Checkbox, Slider

    data = json.loads(Path(json_path).read_text())
    comps: list = []
    mapping: list[tuple[str, str, type]] = []

    # Sort nodes by numeric id for determinism when possible
    def _node_sort(k):
        try:
            return int(k)
        except Exception:
            return k

    for node_id in sorted(data.keys(), key=_node_sort):
        node = data[node_id]
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not inputs:
            continue

        with gr.Accordion(f"{node_id} – {node.get('class_type', '')}", open=False):
            # iterate over inputs in stable order
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
    *dynamic_values: bool | str,
):
    """Function executed by the Gradio UI. Returns tuple of outputs."""
    # Ensure at least 1 run (number of desired successful images)
    batch_runs = max(1, int(batch_runs))

    # 1. Determine selected checks
    checks: List[str] = []
    if do_nsfw:
        checks.append("nsfw")
    if do_face_count:
        checks.append("face_count")
    if do_partial_face:
        checks.append("partial_face")

    if not checks:
        return None, {}, "You must select at least one check.", None, "In Queue: 0"

    output_dir = Path(output_dir_str) if output_dir_str else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = Path(final_dir_str) if final_dir_str else Path.home() / "ComfyUI" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    # Album de imágenes generadas (solo las que superan los checks)
    album_images: List[Image.Image] = []

    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False  # reset at start of each invocation

    # Split dynamic_values into nsfw category toggles and workflow edits
    num_nsfw_cats = len([k for k in CATEGORY_DISPLAY_MAPPINGS.keys() if k != "NOT_DETECTED"])
    nsfw_categories = dynamic_values[:num_nsfw_cats]
    wf_edit_values = dynamic_values[num_nsfw_cats:]

    wf1_edit_vals = wf_edit_values[: len(WF1_MAPPING)]
    wf2_edit_vals = wf_edit_values[len(WF1_MAPPING) :]

    # ----------------------------------------------------------------------
    # Build modified workflow copies according to edits (only once at start
    # of batch). They will be reused across attempts, but seed will still be
    # incremented later as before.
    # ----------------------------------------------------------------------

    wf1_path_mod = _apply_edits_to_workflow(oc.WORKFLOW1_JSON, WF1_MAPPING, wf1_edit_vals)
    wf2_path_mod = _apply_edits_to_workflow(oc.WORKFLOW2_JSON, WF2_MAPPING, wf2_edit_vals)

    target_successes = batch_runs
    success_count = 0
    attempt = 0

    while success_count < target_successes:
        if CANCEL_REQUESTED:
            log_lines.append("[CANCEL] Execution cancelled by user.")
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield None, current_log, None, album_images, f"In Queue: {pending}"
            break

        attempt += 1
        # Separador visual entre intentos
        if attempt > 1:
            log_lines.append("")  # línea en blanco
        log_lines.append(
            f"========== ATTEMPT {attempt} | Passed {success_count}/{target_successes} =========="
        )

        # ------------------------------------------------------------------
        # Run Workflow 1
        # ------------------------------------------------------------------
        log_lines.append("[WF1] Running Workflow1…")
        start_time = time.time()
        try:
            from orchestrator import increment_seed_in_workflow
            new_seed = increment_seed_in_workflow(wf1_path_mod)
            if new_seed is not None:
                log_lines.append(f"[SEED] Updated seed to {new_seed}")
            else:
                log_lines.append("[SEED] Warning: could not update seed (node not found)")
            oc.run_workflow(wf1_path_mod)
        except Exception as e:
            err_msg = f"Error while executing Workflow1: {e}"
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield None, current_log, None, album_images, f"Pendientes: {pending}"
            continue

        wf1_img_path = oc.newest_file(output_dir, after=start_time)
        if wf1_img_path is None:
            err_msg = "Could not find the resulting image from Workflow1."
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield None, current_log, None, album_images, f"Pendientes: {pending}"
            continue

        wf1_img_pil = _load_img(wf1_img_path)

        log_lines.append(f"[WF1] Image generated: {wf1_img_path}")

        # Show WF1 image immediately
        current_log = "\n\n".join(log_lines)
        pending = max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"

        # ----------------------------------------------------------------------
        # Build optional parameters for run_checks according to UI
        # ----------------------------------------------------------------------
        allowed_categories = None
        if do_nsfw and nsfw_categories:
            cat_keys = [k for k in sorted(CATEGORY_DISPLAY_MAPPINGS.keys()) if k != "NOT_DETECTED"]
            allowed_categories = {cat_keys[i] for i, checked in enumerate(nsfw_categories) if checked}

        log_lines.append("[CHECK] Running external checks…")
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
            pending = max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"
            continue

        log_lines.append(f"[CHECK] Results: {json.dumps(results, indent=2, ensure_ascii=False)}")

        current_log = "\n\n".join(log_lines)
        pending = max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"

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
            # La ejecución se considera fallida; no incrementamos success_count
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"
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
        start_time2 = time.time()
        overrides = {oc.LOAD_NODE_ID_WF2: {"image": str(verified_path)}}
        try:
            oc.run_workflow(wf2_path_mod, overrides=overrides)
        except Exception as e:
            err_msg = f"Error while executing Workflow2: {e}"
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"
            continue

        wf2_img_path = oc.newest_file(output_dir, after=start_time2)
        if wf2_img_path is None:
            err_msg = "Could not find the resulting image from Workflow2."
            log_lines.append(err_msg)
            current_log = "\n\n".join(log_lines)
            pending = max(0, target_successes - success_count)
            yield wf1_img_pil, current_log, None, album_images, f"Pendientes: {pending}"
            continue

        wf2_img_pil = _load_img(wf2_img_path)
        log_lines.append(f"[WF2] Image generated: {wf2_img_path}")

        # Add the approved final image to the album and display
        album_images.append(wf2_img_pil)

        current_log = "\n\n".join(log_lines)
        pending = max(0, target_successes - success_count)
        yield wf1_img_pil, current_log, wf2_img_pil, album_images, f"Pendientes: {pending}"

        try:
            final_path = final_dir / wf2_img_path.name
            shutil.copy(wf2_img_path, final_path)
        except Exception as e:
            log_lines.append(f"Warning: could not copy final image to {final_path}: {e}")

        # Si llegamos aquí la imagen pasó todos los filtros ⇒ contamos como éxito
        success_count += 1

        # end loop iteration

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

        run_btn = gr.Button("Run Pipeline", variant="primary")
        pending_box = gr.Markdown("Pendientes: 0", elem_id="pending-box", elem_classes=["pending-box"])

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
                    )

                    margin_slider = gr.Slider(
                        0.0,
                        2.0,
                        0.5,
                        step=0.1,
                        label="Partial face margin",
                    )

                    batch_runs_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=1,
                        step=1,
                        label="Number of runs",
                    )

                    pass  # directory inputs finished

            with gr.Column(scale=3):
                with gr.Row(equal_height=True):
                    wf1_img_out = gr.Image(
                        label="Workflow 1 Image",
                        interactive=False,
                        height=512,
                        elem_classes=["preview-img"],
                    )
                    wf2_img_out = gr.Image(
                        label="Workflow 2 Image",
                        interactive=False,
                        height=512,
                        elem_classes=["preview-img"],
                    )

        with gr.Row():
            log_text = gr.Textbox(label="Log", lines=15, scale=1)

            album_gallery = gr.Gallery(
                label="Results",
                columns=[1, 2, 3, 4],  # responsive: mobile→1, desktop→2-4 columns
                elem_id="results-gallery",
                scale=1,
            )

        # Dynamic workflow editors ------------------------------------------------

        global WF1_MAPPING, WF2_MAPPING

        with gr.Tab("Workflow 1 Config"):
            wf1_components, WF1_MAPPING = _build_workflow_editor(str(oc.WORKFLOW1_JSON))

        with gr.Tab("Workflow 2 Config"):
            wf2_components, WF2_MAPPING = _build_workflow_editor(str(oc.WORKFLOW2_JSON))

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
                batch_runs_slider,
                *nsfw_cat_checkboxes,
                *wf1_components,
                *wf2_components,
            ],
            outputs=[wf1_img_out, log_text, wf2_img_out, album_gallery, pending_box],
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=18188, share=True)


if __name__ == "__main__":
    launch_gui() 