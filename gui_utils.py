"""
Unified utilities for WorkflowUI application.
This module contains all utility functions, component factories, and common operations
previously scattered and duplicated throughout orchestrator_gui.py.
"""

import json
import tempfile
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import gradio as gr
import numpy as np
from PIL import Image

from app_config import CONFIG, AUDIO_CONFIG, WORKFLOW_CONFIG

# =============================================================================
# COMPONENT FACTORY FUNCTIONS
# =============================================================================

def create_number_input(
    label: str, 
    value: Union[int, float, None] = None, 
    precision: int = 0,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None
) -> gr.Number:
    """
    Unified function to create number inputs, replacing duplicated _num and _num2 functions.
    
    Args:
        label: Label for the input
        value: Default value
        precision: Number of decimal places (0 for integers)
        minimum: Minimum allowed value
        maximum: Maximum allowed value
        step: Step size for the input
        
    Returns:
        Gradio Number component
    """
    return gr.Number(
        label=label,
        value=value if value is not None else 0,
        precision=precision,
        minimum=minimum,
        maximum=maximum,
        step=step
    )

def create_textbox_input(
    label: str,
    value: str = "",
    lines: int = 1,
    placeholder: str = "",
    info: str = "",
    interactive: bool = True
) -> gr.Textbox:
    """
    Unified function to create text inputs with consistent styling.
    
    Args:
        label: Label for the input
        value: Default value
        lines: Number of lines (for multiline inputs)
        placeholder: Placeholder text
        info: Information text
        interactive: Whether the input is interactive
        
    Returns:
        Gradio Textbox component
    """
    return gr.Textbox(
        label=label,
        value=value,
        lines=lines,
        placeholder=placeholder,
        info=info,
        interactive=interactive
    )

def create_log_textbox(
    label: str = "",
    lines: int = 12,
    show_label: bool = False
) -> gr.Textbox:
    """
    Create a standardized log textbox with consistent styling.
    
    Args:
        label: Label for the textbox
        lines: Number of lines to display
        show_label: Whether to show the label
        
    Returns:
        Gradio Textbox component with log styling
    """
    return gr.Textbox(
        label=label,
        lines=lines,
        interactive=False,
        show_label=show_label,
        elem_classes=["log-box"]
    )

# =============================================================================
# WORKFLOW UTILITY FUNCTIONS
# =============================================================================

def get_default_node_value(node: Optional[Dict], key: str, fallback: Any = None) -> Any:
    """
    Unified function to extract default values from workflow nodes.
    Replaces the duplicated _def_val functions.
    
    Args:
        node: Node dictionary from workflow JSON
        key: Key to extract from node inputs
        fallback: Default value if key not found
        
    Returns:
        Value from node or fallback
    """
    if node and isinstance(node, dict):
        return node.get("inputs", {}).get(key, fallback)
    return fallback

def load_workflow_data(workflow_path: Union[str, Path]) -> Dict:
    """
    Safely load workflow JSON data with error handling.
    
    Args:
        workflow_path: Path to workflow JSON file
        
    Returns:
        Dictionary containing workflow data, empty dict on error
    """
    try:
        return json.loads(Path(workflow_path).read_text())
    except Exception:
        return {}

def apply_workflow_edits(
    original_path: str, 
    mapping: List[Tuple[str, str, type]], 
    values: List[str]
) -> str:
    """
    Apply user edits to workflow and return path to temporary modified file.
    
    Args:
        original_path: Path to original workflow JSON
        mapping: List of (node_id, input_key, original_type) tuples
        values: List of new values to apply
        
    Returns:
        Path to temporary modified workflow file
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

        # Special case: resolve node 190 file_path relative to the "texts" folder
        if node_id == "190" and key == "file_path":
            try:
                base_dir = Path(__file__).resolve().parent / "texts"
                resolved_path = Path(cast_val)
                if not resolved_path.is_absolute():
                    cast_val = str((base_dir / resolved_path).resolve())
                else:
                    cast_val = str(resolved_path)
            except Exception:
                pass

        if node_id in data and "inputs" in data[node_id]:
            data[node_id]["inputs"][key] = cast_val

    tmp_fd, tmp_path = tempfile.mkstemp(suffix="_workflow.json")
    import os
    os.close(tmp_fd)
    Path(tmp_path).write_text(json.dumps(data, indent=2))
    return tmp_path

def set_workflow_input_by_id(
    path_json: str, 
    node_id: str, 
    key: str, 
    value: Any
) -> bool:
    """
    Set a specific input value in a workflow JSON file.
    
    Args:
        path_json: Path to workflow JSON file
        node_id: ID of the node to modify
        key: Input key to set
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = json.loads(Path(path_json).read_text())
        node = data.get(node_id)
        if isinstance(node, dict):
            node.setdefault("inputs", {})[key] = value
            Path(path_json).write_text(json.dumps(data, indent=2))
            return True
    except Exception:
        pass
    return False

def set_workflow_seed(
    path: str, 
    value: int, 
    node_class: str = WORKFLOW_CONFIG.LOAD_PROMPT_NODE_CLASS
) -> bool:
    """
    Set seed value in workflow for nodes of specified class.
    
    Args:
        path: Path to workflow JSON file
        value: Seed value to set
        node_class: Node class to target
        
    Returns:
        True if successful, False otherwise
    """
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

# =============================================================================
# IMAGE UTILITY FUNCTIONS
# =============================================================================

def load_image_safe(path: Path) -> Optional[Image.Image]:
    """
    Safely load an image file with error handling.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image object or None if loading failed
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

# =============================================================================
# AUDIO UTILITY FUNCTIONS
# =============================================================================

def get_notification_sound() -> Union[str, Tuple[int, np.ndarray]]:
    """
    Get notification sound for Gradio Audio component.
    Returns file path if audio file exists, otherwise generates a beep tone.
    
    Returns:
        Either file path string or tuple of (sample_rate, audio_array)
    """
    if CONFIG.paths.AUDIO_FILE.exists():
        return str(CONFIG.paths.AUDIO_FILE)
    
    # Generate fallback beep tone
    duration = AUDIO_CONFIG.DURATION
    frequency = AUDIO_CONFIG.FREQUENCY
    sample_rate = AUDIO_CONFIG.SAMPLE_RATE
    amplitude = AUDIO_CONFIG.AMPLITUDE
    
    t = np.linspace(0, duration, int(sample_rate * duration), False, dtype=np.float32)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return sample_rate, tone.astype(np.float32)

# =============================================================================
# FILENAME UTILITY FUNCTIONS
# =============================================================================

def sanitize_filename(text: str, max_length: int = 255) -> str:
    """
    Sanitize text for use as filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of resulting filename
        
    Returns:
        Sanitized filename string
    """
    import re
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text.strip()) if text else ""
    return sanitized[:max_length] if sanitized else ""

def extract_filename_base(wf1_json: str, wf2_json: str) -> Optional[str]:
    """
    Extract filename base from workflow configurations.
    
    Args:
        wf1_json: Path to Workflow 1 JSON file
        wf2_json: Path to Workflow 2 JSON file
        
    Returns:
        Filename base string or None if extraction failed
    """
    # Extract prefix from node 190 (Load Prompt From File)
    prefix = None
    try:
        data_wf1 = json.loads(Path(wf1_json).read_text())
        node190 = data_wf1.get("190", {})
        if isinstance(node190.get("inputs"), dict):
            txt_path = node190["inputs"].get("file_path")
            seed_val = node190["inputs"].get("seed")
            if isinstance(txt_path, str) and Path(txt_path).exists() and isinstance(seed_val, int):
                lines = Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
                if lines:
                    idx = seed_val % len(lines)
                    raw_line = lines[idx].strip()
                    import re
                    m = re.match(r"\{\{\{([^}]+)\}\}\}", raw_line)
                    if m:
                        prefix = m.group(1)
    except Exception:
        pass

    # Fallback to node 175 (ShowText) if no prefix found
    if prefix is None:
        try:
            data_wf1 = json.loads(Path(wf1_json).read_text())
            v175 = data_wf1.get("175", {}).get("inputs", {}).get("text_0")
            if isinstance(v175, str) and v175.strip():
                prefix = v175.strip()
        except Exception:
            pass

    # Extract suffix from Workflow 2 SaveImage node
    suffix = None
    delimiter = "_"
    
    try:
        data_wf2 = json.loads(Path(wf2_json).read_text())
        save_node = None
        for n in data_wf2.values():
            if isinstance(n, dict) and isinstance(n.get("class_type"), str):
                if n["class_type"].lower().startswith(WORKFLOW_CONFIG.SAVE_IMAGE_NODE_PREFIX):
                    save_node = n
                    break

        if save_node and isinstance(save_node.get("inputs"), dict):
            inputs_si = save_node["inputs"]

            # Get delimiter override
            if isinstance(inputs_si.get("filename_delimiter"), str):
                delimiter = inputs_si["filename_delimiter"] or "_"

            # Helper to resolve value or reference
            def resolve_value(val):
                if isinstance(val, str):
                    return val
                if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], (str, int)):
                    ref_id = str(val[0])
                    ref_node = data_wf2.get(ref_id, {})
                    if isinstance(ref_node.get("inputs"), dict):
                        cand = (
                            ref_node["inputs"].get("text") or 
                            ref_node["inputs"].get("text_0")
                        )
                        if isinstance(cand, str):
                            return cand
                return None

            suffix = resolve_value(inputs_si.get("filename_suffix"))
    except Exception:
        pass

    # Build final filename base
    parts = []
    if prefix and prefix.strip():
        parts.append(prefix.strip())
    if suffix and suffix.strip():
        parts.append(suffix.strip())

    if not parts:
        return None

    filename_base = delimiter.join(sanitize_filename(p) for p in parts if p.strip())
    return filename_base[:255] if filename_base else None

# =============================================================================
# LORA UTILITY FUNCTIONS
# =============================================================================

def populate_lora_from_directory(
    directory_path: str, 
    max_loras: int = WORKFLOW_CONFIG.MAX_LORA_SLOTS
) -> Tuple[List[str], List[str]]:
    """
    Populate LoRA lists from a directory of .safetensors files.
    
    Args:
        directory_path: Path to directory containing LoRA files
        max_loras: Maximum number of LoRAs to select (per node)
        
    Returns:
        Tuple of (lora_244_paths, lora_307_paths)
    """
    lora_244_paths = [""] * max_loras
    lora_307_paths = [""] * max_loras
    
    try:
        dir_path = Path(directory_path).expanduser().resolve()
        if not dir_path.is_dir():
            return lora_244_paths, lora_307_paths

        # Find all .safetensors files
        all_files = sorted(dir_path.rglob("*.safetensors"))
        if not all_files:
            return lora_244_paths, lora_307_paths

        # Find flux realism file for slot 1
        flux_file = None
        for p in all_files:
            if p.name.lower() == CONFIG.prompts.FLUX_REALISM_FILENAME.lower():
                flux_file = p
                break

        # Helper to get relative path from 'lora/' directory
        def get_relative_path(p: Path) -> str:
            parts = list(p.parts)
            if "lora" in parts:
                idx = parts.index("lora")
                rel = Path(*parts[idx+1:])
                return str(rel)
            return p.name

        # Set flux file in first slot if found
        if flux_file is not None:
            rel_flux = get_relative_path(flux_file)
            lora_244_paths[0] = rel_flux
            lora_307_paths[0] = rel_flux

        # Prepare pool excluding flux file
        pool = [p for p in all_files if p != flux_file]
        random.shuffle(pool)

        # Select up to 10 files for distribution
        selected = pool[:10]
        
        # Fill remaining slots
        for i in range(1, max_loras):
            if i - 1 < len(selected):
                lora_244_paths[i] = get_relative_path(selected[i - 1])
        
        for i in range(1, max_loras):
            idx_sel = i - 1 + (max_loras - 1)
            if idx_sel < len(selected):
                lora_307_paths[i] = get_relative_path(selected[idx_sel])

    except Exception:
        pass

    return lora_244_paths, lora_307_paths

# =============================================================================
# PRESET UTILITY FUNCTIONS
# =============================================================================

def save_preset(preset_name: str, config_values: List[Any]) -> bool:
    """
    Save configuration preset to JSON file.
    
    Args:
        preset_name: Name of the preset
        config_values: List of configuration values
        
    Returns:
        True if successful, False otherwise
    """
    try:
        preset_name = preset_name.strip()
        if not preset_name:
            return False
        
        path = CONFIG.paths.PRESET_DIR / f"{preset_name}.json"
        with open(path, "w") as f:
            json.dump(config_values, f)
        return True
    except Exception:
        return False

def load_preset(preset_name: str, default_values: List[Any]) -> List[Any]:
    """
    Load configuration preset from JSON file.
    
    Args:
        preset_name: Name of the preset to load
        default_values: Default values to return if loading fails
        
    Returns:
        List of configuration values
    """
    if not preset_name:
        return default_values
    
    try:
        path = CONFIG.paths.PRESET_DIR / f"{preset_name}.json"
        if not path.exists():
            return default_values
        
        config_values = json.loads(path.read_text())
        if len(config_values) != len(default_values):
            return default_values
        
        return config_values
    except Exception:
        return default_values

def list_presets() -> List[str]:
    """
    Get list of available preset names.
    
    Returns:
        List of preset names (without .json extension)
    """
    try:
        return sorted([p.stem for p in CONFIG.paths.PRESET_DIR.glob("*.json")])
    except Exception:
        return []

def delete_preset(preset_name: str) -> bool:
    """
    Delete a preset file.
    
    Args:
        preset_name: Name of the preset to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not preset_name:
            return False
        
        path = CONFIG.paths.PRESET_DIR / f"{preset_name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception:
        return False

# =============================================================================
# VALIDATION UTILITY FUNCTIONS
# =============================================================================

def validate_pipeline_config(
    has_nsfw: bool,
    has_face_count: bool, 
    has_partial_face: bool,
    batch_runs: int
) -> Tuple[bool, str]:
    """
    Validate pipeline configuration before execution.
    
    Args:
        has_nsfw: Whether NSFW check is enabled
        has_face_count: Whether face count check is enabled
        has_partial_face: Whether partial face check is enabled
        batch_runs: Number of batch runs requested
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (has_nsfw or has_face_count or has_partial_face):
        return False, "You must select at least one check."
    
    if batch_runs < CONFIG.validation.MIN_BATCH_RUNS:
        return False, f"Batch runs must be at least {CONFIG.validation.MIN_BATCH_RUNS}."
    
    return True, ""

# =============================================================================
# CALLBACK UTILITY FUNCTIONS
# =============================================================================

def create_visibility_toggle_callback(condition_fn):
    """
    Create a callback function for toggling component visibility.
    
    Args:
        condition_fn: Function that takes input value and returns boolean
        
    Returns:
        Callback function that returns gr.update with visibility setting
    """
    def callback(value):
        return gr.update(visible=condition_fn(value))
    return callback

def create_choice_update_callback(choices_fn):
    """
    Create a callback function for updating component choices.
    
    Args:
        choices_fn: Function that returns list of choices
        
    Returns:
        Callback function that returns gr.update with new choices
    """
    def callback(*args):
        return gr.update(choices=choices_fn(*args))
    return callback 