import os
import json
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests

# Local imports
from run_checks import ImageChecker, pil2tensor

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Default configuration values. You can override these via CLI flags or
# environment variables.
COMFY_HOST = os.getenv("COMFY_HOST", "http://127.0.0.1:8188")

# -----------------------------------------------------------------------------
# Determine default ComfyUI output directory
# -----------------------------------------------------------------------------

def _detect_default_output_dir() -> str:
    """Return the most plausible ComfyUI *output* directory on this system.

    A fresh Linux/WSL installation de ComfyUI suele guardar las imágenes en
    ``$HOME/ComfyUI/output``. Sin embargo, algunos instaladores (por ejemplo
    la versión "Comfy" empaquetada) añaden un subdirectorio intermedio
    ``Comfy`` quedando como ``$HOME/Comfy/ComfyUI/output``.

    Esta función comprueba ambas variantes y, si ninguna existe todavía,
    devuelve la ruta clásica para no romper compatibilidad.
    """

    candidates = [
        Path.home() / "ComfyUI" / "output",
        Path.home() / "Comfy" / "ComfyUI" / "output",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Fallback to the first candidate (estándar) siquiera exista aún
    return str(candidates[0])

# Usamos variable de entorno si el usuario la define; si no, intentamos detectar
# la ruta correcta automáticamente.
DEFAULT_OUTPUT_DIR = os.getenv("COMFY_OUTPUT_DIR", _detect_default_output_dir())

# -----------------------------------------------------------------------------
# Workflow file locations
# -----------------------------------------------------------------------------

WORKFLOW_DIR = Path(os.getenv("WORKFLOW_DIR", str(Path(__file__).parent / "workflows")))

# Paths to the workflows used by the pipeline (relative to project root)
WORKFLOW1_JSON = str(WORKFLOW_DIR / "Workflow1.json")
WORKFLOW1_PROMPTCONCAT_JSON = str(WORKFLOW_DIR / "Workflow1_PromptConcatenate.json")
# WORKFLOW2_JSON removed - now using WORKFLOW2_ONESTEP_JSON and WORKFLOW2_TWOSTEPS_JSON directly
WORKFLOW2_ONESTEP_JSON = str(WORKFLOW_DIR / "Workflow2_OneStep.json")
WORKFLOW2_TWOSTEPS_JSON = str(WORKFLOW_DIR / "Workflow2_TwoSteps.json")

# Default to TwoSteps for backward compatibility with existing override controls
WORKFLOW2_JSON = WORKFLOW2_TWOSTEPS_JSON

# Node ID from Workflow2 that loads an image from a path. Adjust if you change
# Workflow2.
LOAD_NODE_ID_WF2 = "314"  # "LoadImageFromPath" node

# -----------------------------------------------------------------------------
# Helper functions for talking to ComfyUI REST API
# -----------------------------------------------------------------------------

def _submit_prompt(prompt_json: Dict[str, Any]) -> str:
    """Send a prompt (workflow) to ComfyUI. Returns the prompt_id."""
    resp = requests.post(f"{COMFY_HOST}/prompt", json={"prompt": prompt_json}, timeout=60)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def _queue_empty() -> bool:
    q = requests.get(f"{COMFY_HOST}/queue", timeout=10).json()
    return not q.get("queue_running") and not q.get("queue_pending")


def run_workflow(path: str, overrides: Dict[str, Dict[str, Any]] | None = None) -> str:
    """Run a workflow JSON file and wait until it finishes.

    Parameters
    ----------
    path: str
        Filesystem path to the workflow JSON.
    overrides: dict | None
        Optional mapping ``{node_id: {input_key: value}}`` to patch before sending.

    Returns
    -------
    str
        The prompt_id for reference.
    """
    with open(path, "r", encoding="utf-8") as fh:
        workflow = json.load(fh)

    if overrides:
        for node_id, patch in overrides.items():
            if node_id in workflow and "inputs" in workflow[node_id]:
                workflow[node_id]["inputs"].update(patch)
            else:
                raise KeyError(f"Node {node_id} not found in {path}")

    prompt_id = _submit_prompt(workflow)
    # Wait until queue is empty (no running or pending jobs)
    while not _queue_empty():
        time.sleep(1)
    return prompt_id

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def newest_file(root: Path, pattern: str = "*.png", after: float | None = None) -> Path | None:
    """Return the most recently modified file under *root* matching *pattern*.
    Optionally filter only files modified after *after* (timestamp, epoch seconds)."""
    files = list(root.rglob(pattern))
    if after is not None:
        files = [p for p in files if p.stat().st_mtime > after]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

# -----------------------------------------------------------------------------
# Verification wrapper
# -----------------------------------------------------------------------------

def run_checks(
    image_path: Path,
    checks: List[str],
    models_dir: str = "Nodes",
    *,
    allowed_categories: set[str] | None = None,
    confidence: float = 0.8,
    margin: float = 0.5,
) -> Dict[str, Any]:
    """Run the selected *checks* on the image located at *image_path*.

    Parameters
    ----------
    image_path : Path
        Path to the image that will be analysed.
    checks : list[str]
        List containing any of: ``"nsfw"``, ``"face_count"``, ``"partial_face"``.
    models_dir : str, default "Nodes"
        Directory where facexlib weights are located.
    allowed_categories : set[str] | None, optional
        Sub-set of NudeNet categories that should be considered *NSFW*.
        When *None* the default set in ``run_checks.DEFAULT_NSFW_CATEGORIES`` is used.
    confidence : float, default 0.8
        Minimum face-detection confidence to consider a bounding box valid.
    margin : float, default 0.5
        Safety margin (as a fraction of face bbox size) used when checking
        whether a face is partially outside the frame.

    Returns
    -------
    dict
        Dictionary with the results of each requested check.
    """

    checker = ImageChecker(retinaface_model_path=models_dir)
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_tensor = pil2tensor(img)  # shape: (1,H,W,C) / [0..1]

    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # NSFW Check
    # ------------------------------------------------------------------
    if "nsfw" in checks:
        is_nsfw, cat = checker.check_nsfw(img_tensor, allowed_categories=allowed_categories)
        results["is_nsfw"] = is_nsfw
        results["nsfw_category"] = cat

    # ------------------------------------------------------------------
    # Face-count Check
    # ------------------------------------------------------------------
    if "face_count" in checks:
        results["face_count"] = checker.check_face_count(img_tensor, confidence=confidence)

    # ------------------------------------------------------------------
    # Partial-face Check (only makes sense if at least one face detected)
    # ------------------------------------------------------------------
    if "partial_face" in checks:
        results["is_partial_face"] = checker.check_partial_face(
            img_tensor, confidence=confidence, margin=margin
        )

    return results

# -----------------------------------------------------------------------------
# Main orchestrator logic
# -----------------------------------------------------------------------------

def pipeline(config: Dict[str, Any]):
    # ------------------------------------------------------------------
    # 1. Run Workflow 1 (generation)
    # ------------------------------------------------------------------
    print("[WF1] Running workflow 1 …")
    start_time = time.time()
    run_workflow(WORKFLOW1_JSON)

    # ------------------------------------------------------------------
    # 2. Locate generated image
    # ------------------------------------------------------------------
    output_dir = Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    img_path = newest_file(output_dir, after=start_time)
    if not img_path:
        raise FileNotFoundError(f"No new image found in {output_dir} after running workflow 1.")
    print(f"[CHECK] Found image: {img_path}")

    # ------------------------------------------------------------------
    # 3. Run external checks
    # ------------------------------------------------------------------
    checks: List[str] = config.get("checks", ["nsfw", "face_count", "partial_face"])
    results = run_checks(img_path, checks)
    print("[CHECK] Results:", results)

    # If any check fails (simple example: is_nsfw == True) -> handle fail
    failed = False
    if results.get("is_nsfw"):
        failed = True
    # Extend with more complex logic as needed

    if failed:
        fail_dir = Path(config.get("fail_dir", output_dir / "rejected"))
        fail_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_path), fail_dir / img_path.name)
        print(f"[CHECK] Image failed checks and was moved to {fail_dir}")
        return  # abort pipeline

    # ------------------------------------------------------------------
    # 4. Prepare image for Workflow 2
    # ------------------------------------------------------------------
    verified_path = Path(config.get("verified_path", output_dir / "verified.png"))
    shutil.copy(str(img_path), verified_path)

    # ------------------------------------------------------------------
    # 5. Run Workflow 2 with override for image path
    # ------------------------------------------------------------------
    print("[WF2] Running workflow 2 …")
    overrides = {LOAD_NODE_ID_WF2: {"image": str(verified_path)}}
    run_workflow(WORKFLOW2_JSON, overrides=overrides)
    print("[DONE] Pipeline finished successfully.")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComfyUI pipeline orchestrator with external checks.")
    parser.add_argument("--config", default="config_checks.json", help="JSON file with configuration.")
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = json.loads(Path(args.config).read_text())
    else:
        print(f"Config file {args.config} not found. Using default config.")
        cfg = {}

    pipeline(cfg)

# -----------------------------------------------------------------------------
# Workflow utilities
# -----------------------------------------------------------------------------

def increment_seed_in_workflow(path: str, node_class: str = "Load Prompt From File - EQX") -> int | None:
    """Increment the *seed* value inside the node whose ``class_type`` equals *node_class*.

    Parameters
    ----------
    path : str
        File path to the workflow JSON that will be modified in place.
    node_class : str, default 'Load Prompt From File - EQX'
        Name of the node whose ``inputs.seed`` should be incremented.

    Returns
    -------
    int | None
        The new seed value or *None* if the node/seed was not found.
    """

    data = json.loads(Path(path).read_text())

    target_node = None
    for node_id, node in data.items():
        if isinstance(node, dict) and node.get("class_type") == node_class:
            target_node = node
            break

    if not target_node:
        return None

    inputs = target_node.get("inputs", {})
    seed_val = inputs.get("seed")
    if isinstance(seed_val, int):
        new_seed = seed_val + 1
        inputs["seed"] = new_seed
        Path(path).write_text(json.dumps(data, indent=2))
        return new_seed
    return None 