# =============================================================================
# PROMPTCONCATENATE UTILITIES - Refactored from orchestrator_gui.py  
# =============================================================================

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Global state for PromptConcatenate
PROMPTCONCAT_INCREMENTAL_COUNTERS: Dict[str, int] = {}
PROMPTCONCAT_DEBUG_LOG: List[str] = []

# =============================================================================
# PROMPTCONCATENATE CONFIGURATION PARSING
# =============================================================================

def parse_promptconcat_configs(dynamic_values: tuple, nsfw_categories_len: int, 
                               wf1_mapping_len: int, wf2_mapping_len: int) -> Dict[str, Dict[str, Any]]:
    """Parse PromptConcatenate configurations from dynamic_values."""
    file_configs = {}
    
    if len(dynamic_values) > nsfw_categories_len + wf1_mapping_len + wf2_mapping_len + 2 + 24:
        promptconcat_start = nsfw_categories_len + wf1_mapping_len + wf2_mapping_len + 2 + 24
        promptconcat_configs = dynamic_values[promptconcat_start:]
        
        try:
            txt_base_dir = Path(__file__).resolve().parent / "txt"
            files_order = []
            for subdir in ["positive", "negative"]:
                subdir_path = txt_base_dir / subdir
                if subdir_path.exists():
                    for txt_file in sorted(subdir_path.glob("*.txt")):
                        files_order.append((subdir, txt_file.name))
            
            config_idx = 0
            for i in range(0, len(promptconcat_configs), 2):
                if i + 1 < len(promptconcat_configs) and config_idx < len(files_order):
                    subdir, filename = files_order[config_idx]
                    file_key = f"{subdir}_{filename}"
                    mode = str(promptconcat_configs[i])
                    fixed_idx = int(promptconcat_configs[i + 1]) if promptconcat_configs[i + 1] else 0
                    file_configs[file_key] = {"mode": mode, "fixed_index": fixed_idx}
                    config_idx += 1
        except Exception as e:
            # Return empty config on error
            pass
    
    return file_configs

# =============================================================================
# PROMPT GENERATION UTILITIES
# =============================================================================

def generate_prompt_preview(subdir: str, joiner: str, file_configs: Dict[str, Dict[str, Any]]) -> str:
    """Generate a preview of what the prompt will look like."""
    try:
        txt_base_dir = Path(__file__).resolve().parent / "txt"
        dir_p = txt_base_dir / subdir
        if not dir_p.exists():
            return ""
        
        parts_files = sorted(dir_p.glob("*.txt"))
        if not parts_files:
            return ""
        
        tokens = []
        for f in parts_files:
            try:
                lines = [ln.strip() for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                if not lines:
                    continue
                
                file_key = f"{subdir}_{f.name}"
                config = file_configs.get(file_key, {})
                mode = config.get("mode", "default")
                
                if mode == "fixed":
                    fixed_idx = config.get("fixed_index", 0)
                    idx = fixed_idx % len(lines)
                elif mode == "incremental":
                    # For preview, just show first line
                    idx = 0
                elif mode == "randomized":
                    # For preview, just show first line
                    idx = 0
                else:
                    # Default mode - use first line for preview
                    idx = 0
                
                tokens.append(lines[idx])
            except Exception:
                continue
        
        return joiner.join(tokens) if tokens else ""
    except Exception:
        return ""

def generate_dynamic_prompt(subdir: str, joiner: str, file_configs: Dict[str, Dict[str, Any]], 
                           current_seed_val: int, attempt: int) -> str:
    """Generate the actual prompt for this execution using current configs."""
    global PROMPTCONCAT_INCREMENTAL_COUNTERS, PROMPTCONCAT_DEBUG_LOG
    
    try:
        txt_base_dir = Path(__file__).resolve().parent / "txt"
        dir_p = txt_base_dir / subdir
        if not dir_p.exists():
            return ""
        
        parts_files = sorted(dir_p.glob("*.txt"))
        if not parts_files:
            return ""
        
        tokens = []
        debug_info = []

        for f in parts_files:
            try:
                lines = [ln.strip() for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                if not lines:
                    continue
                
                file_key = f"{subdir}_{f.name}"
                config = file_configs.get(file_key, {})
                mode = config.get("mode", "default")
                
                if mode == "fixed" or mode == "index":
                    fixed_idx = config.get("fixed_index", 0)
                    idx = fixed_idx % len(lines)
                    mode_label = "INDEX" if mode == "index" else "FIXED"
                    debug_info.append(f"[{mode_label}] {f.name}: index {idx} -> '{lines[idx]}'")
                elif mode == "incremental":
                    # Incremental mode: cycle through lines
                    if file_key not in PROMPTCONCAT_INCREMENTAL_COUNTERS:
                        PROMPTCONCAT_INCREMENTAL_COUNTERS[file_key] = 0
                    idx = PROMPTCONCAT_INCREMENTAL_COUNTERS[file_key] % len(lines)
                    counter_before = PROMPTCONCAT_INCREMENTAL_COUNTERS[file_key]
                    PROMPTCONCAT_INCREMENTAL_COUNTERS[file_key] += 1
                    debug_info.append(f"[INCR] {f.name}: counter {counter_before} -> index {idx} -> '{lines[idx]}'")
                elif mode == "randomized":
                    # Randomized mode: true random selection
                    idx = random.randint(0, len(lines) - 1)
                    debug_info.append(f"[RAND] {f.name}: index {idx} -> '{lines[idx]}'")
                else:
                    # Default mode: seed-based deterministic selection
                    digest = hashlib.sha256((f.name + str(current_seed_val)).encode()).digest()
                    sub_seed = int.from_bytes(digest[:8], "big")
                    rng = random.Random(sub_seed)
                    idx = rng.randrange(len(lines))
                    debug_info.append(f"[DEFT] {f.name}: seed {sub_seed} -> index {idx} -> '{lines[idx]}'")

                tokens.append(lines[idx])
            except Exception as e:
                debug_info.append(f"[ERROR] {f.name}: {str(e)}")

        # Add debug info to global log
        if debug_info:
            PROMPTCONCAT_DEBUG_LOG.extend([f"=== {subdir.upper()} PROMPTS (Attempt {attempt}) ==="] + debug_info)

        return joiner.join(tokens) if tokens else ""
    except Exception as e:
        PROMPTCONCAT_DEBUG_LOG.append(f"[ERROR] {subdir}: {str(e)}")
        return ""

# =============================================================================
# PROMPT INJECTION UTILITIES
# =============================================================================

def inject_promptconcat_preview(wf1_path_mod: str, file_configs: Dict[str, Dict[str, Any]], 
                               log_lines: List[str]) -> None:
    """Inject PromptConcatenate preview prompts into workflow nodes."""
    try:
        data_local = json.loads(Path(wf1_path_mod).read_text())
        
        preview_pos = generate_prompt_preview("positive", " ", file_configs)
        preview_neg = generate_prompt_preview("negative", ", ", file_configs)
        
        # Inject previews into nodes 287 and 288
        if "287" in data_local:
            data_local["287"].setdefault("inputs", {})["text"] = preview_pos
        if "288" in data_local:
            data_local["288"].setdefault("inputs", {})["text"] = preview_neg
        
        Path(wf1_path_mod).write_text(json.dumps(data_local, indent=2))
        log_lines.extend([f"[PROMPTCONCAT] Injected preview prompts into workflow nodes"])
        
    except Exception as e:
        log_lines.extend([f"[WARN] Could not inject PromptConcat prompts: {e}"])

def inject_dynamic_prompts(wf1_path_mod: str, file_configs: Dict[str, Dict[str, Any]], 
                          current_seed_val: int, attempt: int) -> Tuple[str, str]:
    """Inject dynamic PromptConcatenate prompts and return them."""
    global PROMPTCONCAT_DEBUG_LOG
    
    # Clear previous debug log for this attempt
    PROMPTCONCAT_DEBUG_LOG.clear()
    
    # Generate actual prompts for this attempt
    dynamic_pos = generate_dynamic_prompt("positive", " ", file_configs, current_seed_val, attempt)
    dynamic_neg = generate_dynamic_prompt("negative", ", ", file_configs, current_seed_val, attempt)
    
    # Add complete prompts to debug log
    add_complete_prompts_to_debug_log(dynamic_pos, dynamic_neg)
    
    # Inject them into the workflow
    try:
        data_wf1 = json.loads(Path(wf1_path_mod).read_text())
        if "287" in data_wf1:
            data_wf1["287"].setdefault("inputs", {})["text"] = dynamic_pos
        if "288" in data_wf1:
            data_wf1["288"].setdefault("inputs", {})["text"] = dynamic_neg
        
        Path(wf1_path_mod).write_text(json.dumps(data_wf1, indent=2))
    except Exception:
        pass  # Continue if injection fails
    
    return dynamic_pos, dynamic_neg

# =============================================================================
# DEBUG LOG UTILITIES
# =============================================================================

def add_complete_prompts_to_debug_log(dynamic_pos: str, dynamic_neg: str) -> None:
    """Add complete prompts to the debug log with formatting."""
    global PROMPTCONCAT_DEBUG_LOG
    
    PROMPTCONCAT_DEBUG_LOG.extend([
        "",
        "===============================================================================",
        "                              COMPLETE PROMPTS",
        "===============================================================================",
        ""
    ])
    
    # Positive prompt
    if dynamic_pos:
        PROMPTCONCAT_DEBUG_LOG.extend([
            f"POSITIVE PROMPT ({len(dynamic_pos)} characters):",
            "-------------------------------------------------------------------------------"
        ])
        # Split prompt into lines that fit in the box
        prompt_lines = split_prompt_for_display(dynamic_pos)
        for line in prompt_lines:
            PROMPTCONCAT_DEBUG_LOG.append(line)
        
        PROMPTCONCAT_DEBUG_LOG.extend([
            "-------------------------------------------------------------------------------",
            ""
        ])
    else:
        PROMPTCONCAT_DEBUG_LOG.extend([
            "POSITIVE PROMPT: (empty)",
            ""
        ])
    
    # Negative prompt
    if dynamic_neg:
        PROMPTCONCAT_DEBUG_LOG.extend([
            f"NEGATIVE PROMPT ({len(dynamic_neg)} characters):",
            "-------------------------------------------------------------------------------"
        ])
        prompt_lines = split_prompt_for_display(dynamic_neg)
        for line in prompt_lines:
            PROMPTCONCAT_DEBUG_LOG.append(line)
        
        PROMPTCONCAT_DEBUG_LOG.extend([
            "-------------------------------------------------------------------------------",
            ""
        ])
    else:
        PROMPTCONCAT_DEBUG_LOG.extend([
            "NEGATIVE PROMPT: (empty)",
            ""
        ])

def split_prompt_for_display(prompt: str, max_width: int = 75) -> List[str]:
    """Split prompt into lines that fit in the display box."""
    prompt_lines = []
    words = prompt.split()
    current_line = ""
    
    for word in words:
        if len(current_line + word + " ") <= max_width:
            current_line += word + " "
        else:
            if current_line:
                prompt_lines.append(current_line.rstrip())
            current_line = word + " "
    
    if current_line:
        prompt_lines.append(current_line.rstrip())
    
    return prompt_lines

def get_promptconcat_debug_str() -> str:
    """Get the formatted PromptConcatenate debug string."""
    global PROMPTCONCAT_DEBUG_LOG
    if PROMPTCONCAT_DEBUG_LOG:
        return "\n".join(PROMPTCONCAT_DEBUG_LOG)
    else:
        return "No PromptConcatenate debug information available yet."

def reset_promptconcat_counters() -> str:
    """Reset PromptConcatenate incremental counters."""
    global PROMPTCONCAT_INCREMENTAL_COUNTERS
    PROMPTCONCAT_INCREMENTAL_COUNTERS.clear()
    return "PromptConcatenate incremental counters have been reset."

# =============================================================================
# WORKFLOW INTEGRATION UTILITIES
# =============================================================================

def setup_characteristics_text(wf1_path_mod: str, characteristics_text: str, log_lines: List[str]) -> None:
    """Setup characteristics text in the workflow if provided."""
    if characteristics_text and characteristics_text.strip():
        try:
            data_local = json.loads(Path(wf1_path_mod).read_text())
            if "171" in data_local:
                data_local["171"].setdefault("inputs", {})["text"] = characteristics_text.strip()
                Path(wf1_path_mod).write_text(json.dumps(data_local, indent=2))
        except Exception as e:
            log_lines.extend([f"[WARN] Could not set characteristics text for node 171: {e}"])

def is_promptconcat_mode(wf1_mode: str) -> bool:
    """Check if we're in PromptConcatenate mode."""
    return wf1_mode == "PromptConcatenate" 