import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
from PIL import Image

# Local modules
import orchestrator as oc  # our previously created orchestrator.py

# Constants
DEFAULT_OUTPUT_DIR = Path(oc.DEFAULT_OUTPUT_DIR)

# Extra constants for advanced check configuration
from run_checks import CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES

# -----------------------------------------------------------------------------
# Custom CSS to show full portrait images without cropping (object-fit: contain)
# -----------------------------------------------------------------------------

CSS = """
+.preview-img img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
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
    *nsfw_categories: bool,
):
    """Function executed by the Gradio UI. Returns tuple of outputs."""
    # 1. Determine selected checks
    checks: List[str] = []
    if do_nsfw:
        checks.append("nsfw")
    if do_face_count:
        checks.append("face_count")
    if do_partial_face:
        checks.append("partial_face")

    if not checks:
        return None, {}, "Debe seleccionar al menos un check.", None

    output_dir = Path(output_dir_str) if output_dir_str else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    # ----------------------------------------------------------------------
    # Run Workflow 1
    # ----------------------------------------------------------------------
    log_lines.append("[WF1] Running Workflow1…")
    start_time = time.time()
    try:
        oc.run_workflow(oc.WORKFLOW1_JSON)
    except Exception as e:
        return None, {}, f"Error while executing Workflow1: {e}", None

    wf1_img_path = oc.newest_file(output_dir, after=start_time)
    if wf1_img_path is None:
        return None, {}, "Could not find the resulting image from Workflow1.", None

    wf1_img_pil = _load_img(wf1_img_path)

    log_lines.append(f"[WF1] Image generated: {wf1_img_path}")

    # --------------------------------------------------------------
    # Yield early so the user can already see the WF1 image
    # --------------------------------------------------------------
    yield wf1_img_pil, {}, "\n".join(log_lines), None

    # ----------------------------------------------------------------------
    # Build optional parameters for run_checks according to UI
    # ----------------------------------------------------------------------
    allowed_categories = None
    if do_nsfw and nsfw_categories:
        # Map boolean list back to category names (sorted to match UI order)
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
        yield wf1_img_pil, {}, f"Error while executing checks: {e}", None
        return

    log_lines.append(f"[CHECK] Results: {json.dumps(results, indent=2, ensure_ascii=False)}")

    # Emit after checks so the user sees results even si el pipeline va a continuar
    yield wf1_img_pil, results, "\n".join(log_lines), None

    # -----------------------------
    # Decidir si falló
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
        # Devolver solo wf1 image y log
        yield wf1_img_pil, results, "\n".join(log_lines), None
        return

    # ----------------------------------------------------------------------
    # Run Workflow 2
    # ----------------------------------------------------------------------
    verified_path = output_dir / "verified_gui.png"
    try:
        Path(wf1_img_path).replace(verified_path)
    except Exception:
        # if move fails, try copy
        import shutil
        shutil.copy(wf1_img_path, verified_path)

    log_lines.append("[WF2] Running Workflow2…")
    start_time2 = time.time()
    overrides = {oc.LOAD_NODE_ID_WF2: {"image": str(verified_path)}}
    try:
        oc.run_workflow(oc.WORKFLOW2_JSON, overrides=overrides)
    except Exception as e:
        log_lines.append(f"Error while executing Workflow2: {e}")
        yield wf1_img_pil, results, "\n".join(log_lines), None
        return

    wf2_img_path = oc.newest_file(output_dir, after=start_time2)
    if wf2_img_path is None:
        log_lines.append("Could not find the resulting image from Workflow2.")
        yield wf1_img_pil, results, "\n".join(log_lines), None
        return

    wf2_img_pil = _load_img(wf2_img_path)
    log_lines.append(f"[WF2] Image generated: {wf2_img_path}")

    yield wf1_img_pil, results, "\n".join(log_lines), wf2_img_pil
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

        with gr.Row():
            with gr.Column(scale=1):
                output_dir_input = gr.Textbox(label="ComfyUI Output Folder", value=str(DEFAULT_OUTPUT_DIR))

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

                    run_btn = gr.Button("Run Pipeline")

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
            results_json = gr.JSON(label="Check Results")
        log_text = gr.Textbox(label="Log", lines=15)

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
                *nsfw_cat_checkboxes,
            ],
            outputs=[wf1_img_out, results_json, log_text, wf2_img_out],
        )

    demo.queue().launch()


if __name__ == "__main__":
    launch_gui() 