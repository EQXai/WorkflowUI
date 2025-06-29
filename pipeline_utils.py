# Pipeline utilities extracted from orchestrator_gui.py

import json
import math
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import orchestrator as oc
from app_config import CONFIG
from gui_utils import load_image_safe, get_notification_sound, apply_workflow_edits
from run_checks import CATEGORY_DISPLAY_MAPPINGS

# Global seed counter
GLOBAL_SEED_COUNTER: Optional[int] = None

def parse_dynamic_values(dynamic_values: tuple, num_nsfw_cats: int, wf1_mapping_len: int, wf2_mapping_len: int) -> dict:
    """Parse the dynamic_values tuple into organized components."""
    nsfw_categories = dynamic_values[:num_nsfw_cats]
    wf_edit_values = dynamic_values[num_nsfw_cats:]
    wf1_edit_vals = wf_edit_values[:wf1_mapping_len]
    wf2_edit_vals = wf_edit_values[wf1_mapping_len:wf1_mapping_len + wf2_mapping_len]
    
    extra_start = wf1_mapping_len + wf2_mapping_len
    auto_lora_flag = False
    auto_lora_dir = ""
    
    if len(wf_edit_values) >= extra_start + 2:
        auto_lora_flag = bool(wf_edit_values[extra_start])
        auto_lora_dir = str(wf_edit_values[extra_start + 1]).strip()
    
    remaining_vals = wf_edit_values[extra_start + 2:]
    lora244_paths = [""] * 6
    lora244_strengths = [0.7, 0.3, 0.3, 0.3, 0.3, 0.3]
    lora307_paths = [""] * 6
    lora307_strengths = [0.7, 0.3, 0.3, 0.3, 0.3, 0.3]
    
    if remaining_vals and len(remaining_vals) >= 24:
        idx = 0
        lora244_paths = [str(v).strip() for v in remaining_vals[idx:idx+6]]; idx += 6
        lora244_strengths = [float(v) for v in remaining_vals[idx:idx+6]]; idx += 6
        lora307_paths = [str(v).strip() for v in remaining_vals[idx:idx+6]]; idx += 6
        lora307_strengths = [float(v) for v in remaining_vals[idx:idx+6]]
    
    return {
        'nsfw_categories': nsfw_categories,
        'wf1_edit_vals': wf1_edit_vals,
        'wf2_edit_vals': wf2_edit_vals,
        'auto_lora_flag': auto_lora_flag,
        'auto_lora_dir': auto_lora_dir,
        'lora244_paths': lora244_paths,
        'lora244_strengths': lora244_strengths,
        'lora307_paths': lora307_paths,
        'lora307_strengths': lora307_strengths
    }

def process_auto_lora(auto_lora_flag: bool, auto_lora_dir: str, 
                     lora244_paths: List[str], lora244_strengths: List[float],
                     lora307_paths: List[str], lora307_strengths: List[float]) -> Tuple[List[str], List[str], bool, bool]:
    """Process auto-fill LoRA lists from directory if requested."""
    override_loras_244_flag = any(p for p in lora244_paths)
    override_loras_307_flag = any(p for p in lora307_paths)
    
    if auto_lora_flag and auto_lora_dir:
        try:
            dir_path = Path(auto_lora_dir).expanduser().resolve()
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            all_files = sorted(dir_path.rglob("*.safetensors"))
            if not all_files:
                raise FileNotFoundError("No .safetensors files found in directory.")
            
            flux_file = None
            for p in all_files:
                if p.name.lower() == "flux_realism_lora.safetensors":
                    flux_file = p
                    break
            
            def _rel_from_lora(p: Path) -> str:
                parts = list(p.parts)
                if "lora" in parts:
                    idx = parts.index("lora")
                    rel = Path(*parts[idx+1:])
                    return str(rel)
                return p.name
            
            if flux_file is not None:
                rel_flux = _rel_from_lora(flux_file)
                lora244_paths[0] = rel_flux
                lora307_paths[0] = rel_flux
            
            pool = [p for p in all_files if p != flux_file]
            random.shuffle(pool)
            selected = pool[:10]
            
            for i in range(5):
                if i < len(selected):
                    lora244_paths[i+1] = _rel_from_lora(selected[i])
            for i in range(5):
                idx_sel = i + 5
                if idx_sel < len(selected):
                    lora307_paths[i+1] = _rel_from_lora(selected[idx_sel])
            
            override_loras_244_flag = True
            override_loras_307_flag = True
        except Exception:
            pass
    
    return lora244_paths, lora307_paths, override_loras_244_flag, override_loras_307_flag

def create_console_logger():
    """Create an advanced console logging system with formatting."""
    def _fmt_console(line: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        if line.startswith("========== ATTEMPT"):
            return f"\n{ts}  {line}\n"
        for tag in ("[WF1]", "[WF2]", "[CHECK]", "[FINAL]", "[SEED]", "[CANCEL]", "[WARN]", "[CLEANUP]"):
            if line.startswith(tag):
                return f"{ts}  {line}"
        return f"{ts}    {line}"
    
    class _ConsoleList(list):
        def _print(self, txt: str):
            try:
                if not hasattr(self, "_first_print_done"):
                    self._first_print_done = True
                else:
                    stripped = txt.lstrip()
                    if not (stripped.startswith("{") or stripped.startswith("}") or stripped.startswith("\"")):
                        print("", flush=False)
                print(_fmt_console(txt), flush=True)
            except Exception:
                pass
        
        def append(self, item):
            super().append(item)
            self._print(item)
        
        def extend(self, items):
            super().extend(items)
            for _i in items:
                self._print(_i)
    
    return _ConsoleList()

def setup_temporary_prompt_file(load_prompts_directly: bool, prompt_list_str: str) -> Optional[str]:
    """Setup temporary prompt file if direct prompt loading is enabled."""
    if not load_prompts_directly:
        return None
    
    prompt_list_clean = (prompt_list_str or "").strip()
    if not prompt_list_clean:
        raise ValueError("Prompt list is empty while the direct prompt option is enabled.")
    
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix="_prompts.txt")
        os.close(tmp_fd)
        Path(tmp_path).write_text(prompt_list_clean)
        return tmp_path
    except Exception as e:
        raise RuntimeError(f"Could not create temporary prompt file: {e}")

def prepare_workflows(is_promptconcat: bool, wf1_edit_vals: List, wf2_edit_vals: List, 
                     wf1_mapping: List, wf2_mapping: List) -> Tuple[str, str]:
    """Prepare modified workflow copies according to edits."""
    if is_promptconcat:
        wf1_original_path = oc.WORKFLOW1_PROMPTCONCAT_JSON
    else:
        wf1_original_path = oc.WORKFLOW1_JSON
    
    wf1_path_mod = apply_workflow_edits(wf1_original_path, wf1_mapping, wf1_edit_vals)
    wf2_path_mod = apply_workflow_edits(oc.WORKFLOW2_JSON, wf2_mapping, wf2_edit_vals)
    
    return wf1_path_mod, wf2_path_mod

def apply_lora_overrides(wf2_path_mod: str, override_loras_244_flag: bool, override_loras_307_flag: bool,
                        lora244_paths: List[str], lora244_strengths: List[float],
                        lora307_paths: List[str], lora307_strengths: List[float], 
                        log_lines: List[str]) -> None:
    """Apply LoRA overrides if the user enabled either panel."""
    if not (override_loras_244_flag or override_loras_307_flag):
        return
    
    try:
        data_wf2 = json.loads(Path(wf2_path_mod).read_text())
        
        def _patch_node(node_id: str, paths: List[str], strengths: List[float]):
            node = data_wf2.get(node_id)
            if not (node and isinstance(node, dict)):
                return
            for i in range(1, 7):
                path_val = paths[i - 1]
                strength_val = strengths[i - 1]
                if path_val:
                    node.setdefault("inputs", {})[f"lora_{i}"] = {
                        "on": True,
                        "lora": path_val,
                        "strength": strength_val,
                    }
        
        if override_loras_244_flag:
            _patch_node("244", lora244_paths, lora244_strengths)
        
        if override_loras_307_flag:
            _patch_node("307", lora307_paths, lora307_strengths)
        
        try:
            if "249" in data_wf2:
                data_wf2["249"].setdefault("inputs", {})["text"] = ""
        except Exception:
            pass
        
        Path(wf2_path_mod).write_text(json.dumps(data_wf2, indent=2))
    except Exception as e:
        log_lines.extend([f"[WARN] Could not apply LoRA overrides: {e}"])

def set_prompt_file(path_json: str, file_path: str, node_class: str = "Load Prompt From File - EQX") -> bool:
    """Set prompt file path in workflow."""
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

def set_input_by_id(path_json: str, node_id: str, key: str, value):
    """Patch key in node_id inside workflow JSON."""
    try:
        data_local = json.loads(Path(path_json).read_text())
        node = data_local.get(node_id)
        if isinstance(node, dict):
            node.setdefault("inputs", {})[key] = value
            Path(path_json).write_text(json.dumps(data_local, indent=2))
            return True
    except Exception:
        pass
    return False

def generate_seeds(seed_mode: str, seed_counter_input: int) -> dict:
    """Generate all seed values needed for this attempt."""
    global GLOBAL_SEED_COUNTER
    
    current_seed_val: Optional[int] = None
    if seed_mode.lower().startswith("incre"):
        if GLOBAL_SEED_COUNTER is None:
            GLOBAL_SEED_COUNTER = int(seed_counter_input) % (CONFIG.seed.COMFY_MAX_SEED + 1)
        current_seed_val = GLOBAL_SEED_COUNTER
        GLOBAL_SEED_COUNTER = (GLOBAL_SEED_COUNTER + 1) % (CONFIG.seed.COMFY_MAX_SEED + 1)
    elif seed_mode.lower().startswith("rand"):
        current_seed_val = random.randint(0, CONFIG.seed.COMFY_MAX_SEED)
    else:
        current_seed_val = min(int(seed_counter_input), CONFIG.seed.COMFY_MAX_SEED)
    
    image_seed_val = random.randint(0, CONFIG.seed.COMFY_MAX_SEED)
    ksampler2_seed_val = random.randint(0, CONFIG.seed.COMFY_MAX_SEED)
    noise_seed_231 = random.randint(0, CONFIG.seed.COMFY_MAX_SEED)
    noise_seed_294 = random.randint(0, CONFIG.seed.COMFY_MAX_SEED)
    
    return {
        'current_seed': current_seed_val,
        'image_seed': image_seed_val,
        'ksampler2_seed': ksampler2_seed_val,
        'noise_seed_231': noise_seed_231,
        'noise_seed_294': noise_seed_294
    }

def setup_workflow_seeds(wf1_path_mod: str, wf2_path_mod: str, seed_values: dict, log_lines: List[str]) -> None:
    """Apply seeds to workflows."""
    # Note: file_path for node 190 is set earlier (if load_prompts_directly)
    # Do NOT overwrite it here, otherwise it will point to the workflow JSON path
    
    if not set_input_by_id(wf1_path_mod, "189", "seed", seed_values['image_seed']):
        log_lines.append("[WARN] Could not set Seed_KSampler_1 seed (node 189)")
    
    # Set seed for Prompt Loader node 190
    if not set_input_by_id(wf1_path_mod, "190", "seed", seed_values['current_seed']):
        log_lines.append("[WARN] Could not set Prompt Loader seed (node 190)")
    
    if not set_input_by_id(wf1_path_mod, "285", "seed", seed_values['ksampler2_seed']):
        log_lines.append("[WARN] Could not set Seed_KSampler_2 seed (node 285)")
    
    set_input_by_id(wf2_path_mod, "231", "noise_seed", seed_values['noise_seed_231'])
    set_input_by_id(wf2_path_mod, "294", "noise_seed", seed_values['noise_seed_294'])

def extract_prompts_normal(path_json: str, seed_val: int) -> Tuple[Optional[str], Optional[str]]:
    """Extract positive and negative prompts using LoadPromptFromFileEQXNode logic (node 190)."""
    import os
    import re
    
    try:
        data_local = json.loads(Path(path_json).read_text())
        
        # Find node 190 (LoadPromptFromFileEQXNode)
        node190 = data_local.get("190", {})
        if not isinstance(node190, dict):
            return None, None
        
        inputs = node190.get("inputs", {})
        file_path = inputs.get("file_path", "")
        node_seed = inputs.get("seed", 0)
        
        if not file_path:
            return None, None
        
        # Handle relative paths - make them relative to the project root
        if not os.path.isabs(file_path):
            # Try relative to texts/ directory first
            full_path = Path(__file__).resolve().parent / "texts" / file_path
            if not full_path.exists():
                # Try relative to project root
                full_path = Path(__file__).resolve().parent / file_path
        else:
            full_path = Path(file_path)
        
        # Check if file exists
        if not full_path.exists():
            return None, None
        
        # Read the file content
        with open(full_path, 'r', encoding='utf-8') as file:
            data = file.read()
        
        # Use the same regex pattern as LoadPromptFromFileEQXNode
        pattern = r"\{\{\{(.*?)\}\}\}\{\{(.*?)\}\}\{([^}]*)\}"
        matches = re.finditer(pattern, data)
        prompt_list = []
        
        for match in matches:
            prompt_list.append(match.groups())
        
        if not prompt_list:
            return None, None
        
        # Use the seed from the node to select the correct prompt
        # Note: Using node_seed instead of seed_val parameter to match node behavior
        index = node_seed % len(prompt_list)
        identificador, positive_prompt, negative_prompt = prompt_list[index]
        
        return (
            positive_prompt.strip() if positive_prompt else None,
            negative_prompt.strip() if negative_prompt else None
        )
        
    except Exception as e:
        # For debugging, you could uncomment this line:
        # print(f"Error extracting prompts from LoadPromptFromFileEQXNode: {e}")
        pass
    
    return None, None

def extract_filename_base(wf1_json: str, wf2_json: str, is_promptconcat: bool, 
                         pos_prompt: Optional[str], neg_prompt: Optional[str],
                         override_save_name: bool, save_name_base: str) -> Optional[str]:
    """Build a filename base using several nodes that influence it."""
    def _safe(txt: str) -> str:
        import re
        return re.sub(r'[<>:"/\\|?*\s]+', '_', txt)[:50]
    
    if override_save_name and save_name_base.strip():
        return _safe(save_name_base.strip())
    
    # ------------------------------------------------------------------
    # NEW NAMING SCHEME REQUESTED BY USER
    # ------------------------------------------------------------------

    try:
        data_wf2 = json.loads(Path(wf2_json).read_text())
        suffix_raw = data_wf2.get("248", {}).get("inputs", {}).get("text", "")
        suffix = _safe(str(suffix_raw)) if suffix_raw else None
    except Exception:
        suffix = None

    if is_promptconcat:
        # Mode PromptConcatenate ─ prefix comes from node 170 (Prefix Title)
        try:
            data_wf1 = json.loads(Path(wf1_json).read_text())
            prefix_raw = data_wf1.get("170", {}).get("inputs", {}).get("text", "")
            prefix = _safe(str(prefix_raw)) if prefix_raw else None
        except Exception:
            prefix = None

        parts = [p for p in (prefix, suffix) if p]
        return "_".join(parts) if parts else "generated"

    # ------------------------------ Normal mode -----------------------------
    # Extract identifier from the prompt line chosen by node 190
    try:
        data_wf1 = json.loads(Path(wf1_json).read_text())
        n190 = data_wf1.get("190", {}).get("inputs", {})
        file_path = n190.get("file_path", "")
        node_seed = int(n190.get("seed", 0))

        # Resolve relative path (texts/ or project root)
        if file_path and not os.path.isabs(file_path):
            base_dir = Path(__file__).resolve().parent
            cand1 = base_dir / "texts" / file_path
            file_path_abs = cand1 if cand1.exists() else (base_dir / file_path)
        else:
            file_path_abs = Path(file_path)

        ident = None
        if file_path_abs.exists():
            import re
            data_txt = file_path_abs.read_text(encoding="utf-8", errors="ignore")
            pattern = r"\{\{\{(.*?)\}\}\}\{\{(.*?)\}\}\{([^}]*)\}"
            matches = re.findall(pattern, data_txt)
            if matches:
                idx = node_seed % len(matches)
                ident_raw = matches[idx][0] if isinstance(matches[idx], tuple) else matches[idx]
                ident = _safe(ident_raw)
    except Exception:
        ident = None

    parts = [p for p in (ident, suffix) if p]
    return "_".join(parts) if parts else "generated"

def calculate_target_successes(endless_until_cancel: bool, batch_runs: int) -> float:
    """Calculate the target number of successful images."""
    if endless_until_cancel:
        return math.inf
    else:
        return max(1, int(batch_runs))

def should_continue_batch(endless_until_cancel: bool, success_count: int, target_successes: float, cancel_requested: bool) -> bool:
    """Determine if the batch should continue."""
    if cancel_requested:
        return False
    if endless_until_cancel:
        return True
    return success_count < target_successes

def build_allowed_categories(nsfw_categories: tuple) -> set:
    """Build allowed categories set based on current toggles."""
    allowed_categories = set()
    for cat_key, is_enabled in zip(CATEGORY_DISPLAY_MAPPINGS.keys(), nsfw_categories):
        if cat_key != "NOT_DETECTED" and is_enabled:
            allowed_categories.add(cat_key)
    return allowed_categories

def evaluate_check_results(results: dict, stop_multi_faces: bool, stop_partial_face: bool, log_lines: List[str]) -> bool:
    """Evaluate check results and return True if failed."""
    failed = False
    
    if results.get("is_nsfw"):
        failed = True
        log_lines.extend([f"[CHECK] Image flagged as NSFW. Pipeline stops."])
    
    if stop_multi_faces and results.get("face_count", 0) > 1:
        failed = True
        log_lines.extend([f"[CHECK] More than one face detected and stop option is active."])
    
    if stop_partial_face and results.get("is_partial_face"):
        failed = True
        log_lines.extend([f"[CHECK] Partial face detected and stop option is active."])
    
    return failed 