"""
Centralized configuration for WorkflowUI application.
This module contains all constants, default values, and configuration settings
previously scattered throughout orchestrator_gui.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# SEED CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SeedConfig:
    """Configuration for seed management."""
    MAX_SEED_VALUE: int = 18446744073709551615  # Upper limit for any seed value (uint64 max)
    COMFY_MAX_SEED: int = 0xFFFFFFFF  # ComfyUI 32-bit seed limit (4_294_967_295)

# =============================================================================
# UI CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class UIConfig:
    """Configuration for UI behavior and appearance."""
    MAX_GALLERY_IMAGES: int = 20  # Maximum number of images in gallery
    DEFAULT_LOG_LINES: int = 12  # Default number of lines in log textbox
    DEFAULT_SEED_LOG_LINES: int = 8  # Default lines in seed log
    DEFAULT_PROMPT_LOG_LINES: int = 8  # Default lines in prompt log
    
    # Component dimensions
    PREVIEW_IMAGE_HEIGHT: int = 400
    GALLERY_COLUMNS: List[int] = field(default_factory=lambda: [4])
    
    # Server configuration
    SERVER_NAME: str = "0.0.0.0"
    SERVER_PORT: int = 18188
    SHARE: bool = True
    INBROWSER: bool = True

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio notifications."""
    DURATION: float = 0.6
    FREQUENCY: int = 880
    SAMPLE_RATE: int = 44100
    AMPLITUDE: float = 0.25

# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for workflow processing."""
    # Default node values
    DEFAULT_STEPS: int = 20
    DEFAULT_GUIDANCE: float = 7.0
    DEFAULT_GRAIN_POWER: float = 1.0
    DEFAULT_MAX_SIZE: int = 1024
    DEFAULT_DENOISE: float = 0.2
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    
    # LoRA configuration
    DEFAULT_LORA_STRENGTH_PRIMARY: float = 0.7
    DEFAULT_LORA_STRENGTH_SECONDARY: float = 0.3
    MAX_LORA_SLOTS: int = 6
    
    # Special node IDs
    LOAD_PROMPT_NODE_CLASS: str = "Load Prompt From File - EQX"
    SAVE_IMAGE_NODE_PREFIX: str = "saveimage"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for file and directory paths."""
    # Base directories (will be set during initialization)
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    
    def __post_init__(self):
        """Initialize computed paths after base directory is set."""
        self.PRESET_DIR = self.BASE_DIR / "save_presets"
        self.AUDIO_DIR = self.BASE_DIR / "audio"
        self.TEXTS_DIR = self.BASE_DIR / "texts"
        self.TXT_DIR = self.BASE_DIR / "txt"
        self.NODES_DIR = self.BASE_DIR / "Nodes"
        self.DEBUG_DIR = self.BASE_DIR / "WF_debug"
        
        # Audio file
        self.AUDIO_FILE = self.AUDIO_DIR / "notify.mp3"
        
        # Ensure directories exist
        self.PRESET_DIR.mkdir(parents=True, exist_ok=True)
        self.DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for input validation."""
    MIN_BATCH_RUNS: int = 1
    MIN_CONFIDENCE: float = 0.1
    MAX_CONFIDENCE: float = 1.0
    DEFAULT_CONFIDENCE: float = 0.8
    
    MIN_MARGIN: float = 0.0
    MAX_MARGIN: float = 2.0
    DEFAULT_MARGIN: float = 0.5
    
    # Step limits
    MIN_STEPS: int = 1
    MAX_STEPS: int = 150
    
    # Guidance limits
    MIN_GUIDANCE: float = 1.0
    MAX_GUIDANCE: float = 20.0

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt processing."""
    DEFAULT_POSITIVE_JOINER: str = " "
    DEFAULT_NEGATIVE_JOINER: str = ", "
    DEFAULT_FILENAME_DELIMITER: str = "_"
    
    # Default characteristics text
    DEFAULT_CHARACTERISTICS: str = (
        "A photo of GeegeeDwa1 posing next to a white chair. She has long darkbrown hair "
        "styled in pigtails and very pale skin. She wears a vibrant sexy outfit. Her expression "
        "is a smirk. The background shows a modern apartment that exudes a candid atmosphere."
    )
    
    # File extensions
    PROMPT_FILE_EXTENSION: str = "*.txt"
    LORA_FILE_EXTENSION: str = "*.safetensors"
    
    # Special filenames
    FLUX_REALISM_FILENAME: str = "flux_realism_lora.safetensors"

# =============================================================================
# CSS CONFIGURATION
# =============================================================================

CSS_STYLES = """
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

+.log-box textarea {
    border: none !important;
    background-color: transparent !important;
    resize: vertical !important;
    padding: 4px !important;
    font-family: monospace;
    line-height: 1.2;
    height: auto !important;
    min-height: 220px; /* ~12 lines */
}

+.log-box {
    border: none !important;
    background-color: transparent !important;
}
"""

# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

@dataclass
class AppConfig:
    """Main application configuration combining all sub-configurations."""
    seed: SeedConfig = field(default_factory=SeedConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    
    def __post_init__(self):
        """Initialize any computed values after all fields are set."""
        # Ensure path configuration is properly initialized
        if not hasattr(self.paths, 'PRESET_DIR'):
            self.paths.__post_init__()

# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Single global configuration instance
CONFIG = AppConfig()

# Convenient access to commonly used values
SEED_CONFIG = CONFIG.seed
UI_CONFIG = CONFIG.ui
AUDIO_CONFIG = CONFIG.audio
WORKFLOW_CONFIG = CONFIG.workflow
PATH_CONFIG = CONFIG.paths
VALIDATION_CONFIG = CONFIG.validation
PROMPT_CONFIG = CONFIG.prompts 