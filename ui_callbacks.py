"""
UI Callbacks for WorkflowUI application.
This module contains callback functions that handle UI interactions and events.
"""

import json
import shutil
from pathlib import Path
from typing import Any, List

import gradio as gr
from app_config import CONFIG
from gui_utils import save_preset, load_preset, list_presets, delete_preset

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Testing incremental counter for prompt testing
TEST_INCREMENTAL_COUNTER = {}

# Last test results for applying to main panel
LAST_TEST_RESULTS = {
    'positive_configs': [],
    'negative_configs': [],
    'positive_indices': [],
    'negative_indices': []
}

# =============================================================================
# PRESET MANAGEMENT CALLBACKS
# =============================================================================

def filter_presets(pattern: str) -> List[str]:
    """Filter presets based on search pattern."""
    pattern = (pattern or "").lower().strip()
    names = list_presets()
    if not pattern:
        return names
    return [n for n in names if pattern in n.lower()]

def save_preset_wrapper(*vals):
    """Save preset with configuration values."""
    *cfg_vals, preset_name = vals
    success = save_preset(preset_name, cfg_vals)
    if success:
        return gr.update(choices=list_presets(), value=preset_name)
    return gr.update()

def load_preset_wrapper(selected_name: str, default_values: List[Any]) -> List[Any]:
    """Load preset configuration values."""
    return load_preset(selected_name, default_values)

def delete_preset_wrapper(name: str):
    """Delete a preset and update choices."""
    success = delete_preset(name)
    if success:
        return gr.update(choices=list_presets(), value=None)
    return gr.update()

def rename_preset(old_name: str, new_name: str):
    """Rename a preset."""
    new_name = (new_name or "").strip()
    if not old_name or not new_name:
        return gr.update()
    
    src = CONFIG.paths.PRESET_DIR / f"{old_name}.json"
    dst = CONFIG.paths.PRESET_DIR / f"{new_name}.json"
    
    if src.exists():
        src.rename(dst)
    
    return gr.update(choices=list_presets(), value=new_name)

def export_preset(name: str) -> Path | None:
    """Export a preset file."""
    if not name:
        return None
    return CONFIG.paths.PRESET_DIR / f"{name}.json"

def import_preset(file_obj):
    """Import a preset from uploaded file."""
    if file_obj is None:
        return gr.update()
    
    try:
        dst = CONFIG.paths.PRESET_DIR / Path(file_obj.name).name
        shutil.copy(file_obj.name, dst)
        return gr.update(choices=list_presets(), value=dst.stem)
    except Exception:
        return gr.update()

def update_preset_selector(selected: str | None):
    """Update preset selector with current choices."""
    return gr.update(choices=list_presets(), value=selected)

# =============================================================================
# PIPELINE VALIDATION CALLBACKS
# =============================================================================

def validate_pipeline_settings(
    nsfw: bool, 
    face_count: bool, 
    partial_face: bool, 
    batch_runs_val: int | float | None
):
    """Validate pipeline settings before allowing execution."""
    has_check = nsfw or face_count or partial_face
    ok_batch = (batch_runs_val or 0) >= 1
    return gr.update(interactive=bool(has_check and ok_batch))

# =============================================================================
# LORA MANAGEMENT CALLBACKS
# =============================================================================

def populate_lora_slots(dir_path: str, *_):
    """Populate LoRA slots with files from directory."""
    from pathlib import Path as _P
    import random as _rnd
    
    if not dir_path:
        return [gr.update() for _ in range(12)]

    try:
        base = _P(dir_path).expanduser().resolve()
        if not base.is_dir():
            raise FileNotFoundError

        files = sorted(base.rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError

        flux = None
        for p in files:
            if p.name.lower() == "flux_realism_lora.safetensors":
                flux = p
                break

        def _rel(p: _P):
            try:
                return str(p.relative_to(base))
            except ValueError:
                return p.name

        # Prepare selection pool without flux
        pool = [p for p in files if (flux is None or p.resolve() != flux.resolve()) and "flux_realism_lora" not in p.name.lower()]
        _rnd.shuffle(pool)
        sel = pool[:10]

        # Build list of 12 values (6 for 244, 6 for 307)
        values = [""] * 12
        if flux:
            values[0] = _rel(flux)
            values[6] = _rel(flux)

        # Fill positions 1-5 and 7-11
        for i in range(5):
            if i < len(sel):
                values[i+1] = _rel(sel[i])
        for i in range(5):
            idx = i + 5
            if idx < len(sel):
                values[i+7] = _rel(sel[idx])

        return [gr.update(value=v) for v in values]
    except Exception:
        # On error, leave unchanged
        return [gr.update() for _ in range(12)]

# =============================================================================
# PROMPTCONCATENATE CALLBACKS
# =============================================================================

def reset_promptconcat_counters() -> str:
    """Reset incremental counters for PromptConcatenate."""
    global PROMPTCONCAT_INCREMENTAL_COUNTERS
    try:
        from orchestrator_gui import PROMPTCONCAT_INCREMENTAL_COUNTERS
        PROMPTCONCAT_INCREMENTAL_COUNTERS.clear()
        return "Incremental counters reset successfully"
    except Exception:
        return "Error resetting counters"

def create_visibility_toggle(mode_name: str, target_visible_modes: List[str]):
    """Create a visibility toggle callback for specific modes."""
    def toggle_visibility(mode):
        return gr.update(visible=(mode in target_visible_modes))
    return toggle_visibility

# =============================================================================
# PROMPT TESTING CALLBACKS
# =============================================================================

def generate_test_prompts(seed_input: int, mode_input: str, *test_vals) -> tuple[str, str, str, str]:
    """Generate test prompts using the current testing configuration."""
    import random
    from pathlib import Path
    
    try:
        base_dir = Path(__file__).resolve().parent / "txt"
        
        # File lists in order (must match UI creation order)
        positive_files = ["part1_photo_type.txt", "part2_subject.txt", "part3_pose.txt"]
        negative_files = ["neg1_general.txt", "neg2_anatomy.txt", "neg3_artifacts.txt"]
        
        # Process test values - they come in pairs (mode, index) for each file
        # Order: pos1_mode, pos1_index, pos2_mode, pos2_index, pos3_mode, pos3_index,
        #        neg1_mode, neg1_index, neg2_mode, neg2_index, neg3_mode, neg3_index
        if len(test_vals) < 12:
            return "Error: Insufficient test configuration values", "", "", ""
        
        # Extract modes and indices
        pos_configs = []
        neg_configs = []
        
        # Positive files (first 6 values)
        for i in range(3):
            mode = test_vals[i * 2] if test_vals[i * 2] else "default"
            index = test_vals[i * 2 + 1] if test_vals[i * 2 + 1] is not None else 0
            pos_configs.append((mode, int(index)))
        
        # Negative files (next 6 values)
        for i in range(3):
            mode = test_vals[6 + i * 2] if test_vals[6 + i * 2] else "default"
            index = test_vals[6 + i * 2 + 1] if test_vals[6 + i * 2 + 1] is not None else 0
            neg_configs.append((mode, int(index)))
        
        # Use global incremental counter for testing
        global TEST_INCREMENTAL_COUNTER
        
        def select_line(filename: str, lines: list, mode: str, fixed_index: int, seed: int) -> tuple[str, int]:
            """Select a line based on the mode."""
            if not lines:
                return "", 0
                
            if mode == "fixed" or mode == "index":
                # Use the specified index, wrapped to file length
                idx = fixed_index % len(lines)
                return lines[idx].strip(), idx
            elif mode == "incremental":
                # Use incremental counter for this file
                if filename not in TEST_INCREMENTAL_COUNTER:
                    TEST_INCREMENTAL_COUNTER[filename] = 0
                idx = TEST_INCREMENTAL_COUNTER[filename] % len(lines)
                TEST_INCREMENTAL_COUNTER[filename] += 1
                return lines[idx].strip(), idx
            elif mode == "randomized":
                # True random selection - use current time + filename for maximum randomness
                import time
                random_seed = int(time.time() * 1000000) + hash(filename) + seed
                random.seed(random_seed)
                idx = random.randint(0, len(lines) - 1)
                return lines[idx].strip(), idx
            else:  # "default" or any other mode
                # Seed-based selection
                idx = seed % len(lines)
                return lines[idx].strip(), idx
        
        # Process positive files
        positive_parts = []
        positive_indices = []
        debug_lines = [f"=== POSITIVE FILES (Seed: {seed_input}, Default Mode: {mode_input}) ==="]
        
        for i, (filename, (mode, fixed_idx)) in enumerate(zip(positive_files, pos_configs)):
            file_path = base_dir / "positive" / filename
            try:
                if file_path.exists():
                    lines = file_path.read_text(encoding='utf-8').strip().splitlines()
                    selected_text, selected_idx = select_line(filename, lines, mode, fixed_idx, seed_input)
                    positive_parts.append(selected_text)
                    positive_indices.append(str(selected_idx))  # Solo el número
                    debug_lines.append(f"  {filename}: mode={mode}, idx={fixed_idx if mode in ['fixed', 'index'] else 'auto'} → [{selected_idx}] {selected_text}")
                else:
                    debug_lines.append(f"  {filename}: FILE NOT FOUND")
            except Exception as e:
                debug_lines.append(f"  {filename}: ERROR - {str(e)}")
        
        # Process negative files
        negative_parts = []
        negative_indices = []
        debug_lines.append(f"\n=== NEGATIVE FILES ===")
        
        for i, (filename, (mode, fixed_idx)) in enumerate(zip(negative_files, neg_configs)):
            file_path = base_dir / "negative" / filename
            try:
                if file_path.exists():
                    lines = file_path.read_text(encoding='utf-8').strip().splitlines()
                    selected_text, selected_idx = select_line(filename, lines, mode, fixed_idx, seed_input)
                    negative_parts.append(selected_text)
                    negative_indices.append(str(selected_idx))  # Solo el número
                    debug_lines.append(f"  {filename}: mode={mode}, idx={fixed_idx if mode in ['fixed', 'index'] else 'auto'} → [{selected_idx}] {selected_text}")
                else:
                    debug_lines.append(f"  {filename}: FILE NOT FOUND")
            except Exception as e:
                debug_lines.append(f"  {filename}: ERROR - {str(e)}")
        
        # Combine results
        positive_prompt = " ".join(positive_parts) if positive_parts else ""
        negative_prompt = ", ".join(negative_parts) if negative_parts else ""
        
        # Format indices info - solo números separados por comas
        indices_info = f"POSITIVE: {','.join(positive_indices) if positive_indices else ''}\nNEGATIVE: {','.join(negative_indices) if negative_indices else ''}"
        
        # Format debug info
        debug_info = "\n".join(debug_lines)
        
        # Save results globally for applying to main panel
        global LAST_TEST_RESULTS
        LAST_TEST_RESULTS = {
            'positive_configs': pos_configs,
            'negative_configs': neg_configs,
            'positive_indices': positive_indices,
            'negative_indices': negative_indices
        }
        
        return positive_prompt, negative_prompt, indices_info, debug_info
        
    except Exception as e:
        error_msg = f"Error generating test prompts: {str(e)}"
        return error_msg, "", "", error_msg

def clear_test_results() -> tuple[str, str, str, str]:
    """Clear all test results."""
    global LAST_TEST_RESULTS
    # Clear saved test results
    LAST_TEST_RESULTS = {
        'positive_configs': [],
        'negative_configs': [],
        'positive_indices': [],
        'negative_indices': []
    }
    return "", "", "", ""

def reset_test_counters() -> str:
    """Reset incremental counters for prompt testing."""
    global TEST_INCREMENTAL_COUNTER
    TEST_INCREMENTAL_COUNTER.clear()
    return "Test incremental counters reset successfully"

def apply_test_results_to_main() -> tuple[str, ...]:
    """Apply test results to main panel."""
    global LAST_TEST_RESULTS
    try:
        if not LAST_TEST_RESULTS['positive_configs'] and not LAST_TEST_RESULTS['negative_configs']:
            return "No test results to apply. Please generate test prompts first.", *([gr.update() for _ in range(12)])
        
        # Build updates for promptconcat_components
        # Structure: [pos1_mode, pos1_index, pos2_mode, pos2_index, pos3_mode, pos3_index,
        #            neg1_mode, neg1_index, neg2_mode, neg2_index, neg3_mode, neg3_index]
        updates = []
        
        # Process positive configurations (3 files)
        for i in range(3):
            if i < len(LAST_TEST_RESULTS['positive_indices']):
                # Use "index" mode with the exact index from testing results
                selected_index = int(LAST_TEST_RESULTS['positive_indices'][i])
                updates.append(gr.update(value="index"))  # mode_radio set to "index"
                updates.append(gr.update(value=selected_index, visible=True))  # fixed_index with exact value
            else:
                updates.append(gr.update())  # mode_radio
                updates.append(gr.update())  # fixed_index
        
        # Process negative configurations (3 files)
        for i in range(3):
            if i < len(LAST_TEST_RESULTS['negative_indices']):
                # Use "index" mode with the exact index from testing results
                selected_index = int(LAST_TEST_RESULTS['negative_indices'][i])
                updates.append(gr.update(value="index"))  # mode_radio set to "index"
                updates.append(gr.update(value=selected_index, visible=True))  # fixed_index with exact value
            else:
                updates.append(gr.update())  # mode_radio
                updates.append(gr.update())  # fixed_index
        
        # Ensure we have exactly 12 updates
        while len(updates) < 12:
            updates.append(gr.update())
        
        # Get the indices for display
        pos_indices = ",".join(LAST_TEST_RESULTS['positive_indices']) if LAST_TEST_RESULTS['positive_indices'] else ""
        neg_indices = ",".join(LAST_TEST_RESULTS['negative_indices']) if LAST_TEST_RESULTS['negative_indices'] else ""
        status_msg = f"Test results applied successfully!\nPOSITIVE: {pos_indices}\nNEGATIVE: {neg_indices}"
        
        return status_msg, *updates
        
    except Exception as e:
        empty_updates = [gr.update() for _ in range(12)]
        return f"Error applying test results: {str(e)}", *empty_updates

# =============================================================================
# UTILITY CALLBACKS
# =============================================================================

def show_component_temporarily():
    """Show a component temporarily."""
    return gr.update(visible=True)

def hide_component():
    """Hide a component."""
    return gr.update(visible=False)

def reset_to_defaults(default_values: List[Any]) -> List[Any]:
    """Reset all components to default values."""
    return default_values 