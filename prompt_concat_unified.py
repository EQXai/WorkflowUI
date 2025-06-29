"""
Prompt Concatenate Unified - EQX
================================

A powerful ComfyUI node for building structured prompts from text file collections.
This node provides sophisticated prompt construction with multiple selection strategies,
comprehensive debugging, and flexible concatenation options.

Features:
- Three selection modes: deterministic, incremental, and random
- Support for positive and negative prompt construction
- Comprehensive debugging and logging
- Flexible file organization and validation
- Session-aware incremental counters
- Error handling with graceful fallbacks

Author: EQX
Version: 1.0.0
Category: Prompt/Modular
"""

import os
import random
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class SelectionMode(Enum):
    """Enumeration of available selection modes"""
    DETERMINISTIC = "deterministic"
    INCREMENTAL = "incremental" 
    RANDOM = "random"


DEFAULT_CONFIG = {
    "base_dir": "txt",
    "positive_joiner": " ",
    "negative_joiner": ", ",
    "seed": 42,
    "selection_mode": SelectionMode.DETERMINISTIC.value,
    "encoding": "utf-8",
    "file_extension": "*.txt"
}

REQUIRED_SUBDIRS = ["positive", "negative"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PromptSelectionInfo:
    """
    Detailed information about how a specific prompt was selected.
    Used for debugging and reproducibility tracking.
    """
    file_name: str
    selected_line: str
    selected_index: int
    total_lines: int
    selection_mode: str
    seed_used: int
    file_path: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable representation of selection info"""
        return (f"{self.file_name}: [{self.selected_index}/{self.total_lines-1}] "
                f"'{self.selected_line[:50]}{'...' if len(self.selected_line) > 50 else ''}'")


@dataclass
class DirectoryAnalysis:
    """Analysis results of a prompt directory structure"""
    base_path: Path
    positive_dir: Path
    negative_dir: Path
    positive_files: List[Path] = field(default_factory=list)
    negative_files: List[Path] = field(default_factory=list)
    total_positive_lines: int = 0
    total_negative_lines: int = 0
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileConfig:
    """Configuration for individual file selection behavior"""
    mode: str
    seed: Optional[int] = None
    start_index: Optional[int] = None
    
    def __post_init__(self):
        if self.mode not in [mode.value for mode in SelectionMode]:
            raise ValueError(f"Invalid mode: {self.mode}")


@dataclass 
class GlobalFileConfig:
    """Global configuration for all file selections"""
    file_configs: Dict[str, FileConfig] = field(default_factory=dict)
    default_mode: str = SelectionMode.DETERMINISTIC.value
    default_seed: int = 42
    
    def get_config_for_file(self, file_path: str) -> FileConfig:
        """Get configuration for a specific file, or return default"""
        # Try exact match first
        if file_path in self.file_configs:
            return self.file_configs[file_path]
        
        # Try normalized path (replace backslashes with forward slashes)
        normalized_path = file_path.replace('\\', '/')
        if normalized_path in self.file_configs:
            return self.file_configs[normalized_path]
        
        # Try just filename
        filename = os.path.basename(file_path)
        for config_path, config in self.file_configs.items():
            if os.path.basename(config_path) == filename:
                return config
        
        # Return default configuration
        return FileConfig(
            mode=self.default_mode,
            seed=self.default_seed
        )


@dataclass
class AssemblyResult:
    """Complete result of prompt assembly operation"""
    positive_prompt: str
    negative_prompt: str
    positive_indices: List[int]
    negative_indices: List[int]
    positive_debug: List[PromptSelectionInfo]
    negative_debug: List[PromptSelectionInfo]
    seed_used: int
    selection_mode: str
    file_config_used: Optional[GlobalFileConfig] = None
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# CORE PROMPT CONCATENATION CLASS
# =============================================================================

class PromptConcatenateUnified:
    """
    Unified node that builds positive and negative prompts from text file collections.
    
    This node provides a sophisticated system for constructing prompts by sampling
    lines from organized text files. It's designed for modular prompt engineering
    where different aspects of a prompt (style, subject, mood, etc.) are stored
    in separate files.
    
    Directory Structure Required:
    ```
    base_dir/
    ├── positive/
    │   ├── styles.txt      # One line will be selected from each file
    │   ├── subjects.txt    # Files are processed in alphabetical order
    │   ├── moods.txt       # Empty files are safely ignored
    │   └── ...
    └── negative/
        ├── avoid.txt       # Same structure as positive
        ├── problems.txt
        └── ...
    ```
    
    Selection Modes:
    - **deterministic**: Reproducible selection based on seed (seed % line_count)
    - **incremental**: Cycles through lines sequentially per session
    - **random**: True random selection each execution
    
    Outputs:
    - positive_prompt: Assembled positive prompt string
    - negative_prompt: Assembled negative prompt string  
    - positive_index: Comma-separated indices of selected lines
    - negative_index: Comma-separated indices of selected lines
    - debug_info: Detailed information about selection process
    - seed_used: The seed value that was used
    - help_info: Complete guide on selection modes and usage
    - recovery_config: JSON configuration to reproduce this exact result
    """
    
    # ComfyUI Node Configuration
    NODE_NAME = "Prompt Concatenate Unified - EQX"
    CATEGORY = "Prompt/Modular"
    FUNCTION = "assemble"
    
    def __init__(self):
        """Initialize the node with clean session state"""
        self._session_counters: Dict[str, int] = {}
        self._logger = self._setup_logging()
        self._last_analysis: Optional[DirectoryAnalysis] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the node"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(name)s] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for ComfyUI"""
        return {
            "required": {
                "base_dir": (
                    "STRING", 
                    {
                        "multiline": False, 
                        "default": DEFAULT_CONFIG["base_dir"],
                        "tooltip": "Directory containing 'positive' and 'negative' subfolders"
                    }
                ),
                "positive_joiner": (
                    "STRING", 
                    {
                        "default": DEFAULT_CONFIG["positive_joiner"], 
                        "multiline": False,
                        "tooltip": "String used to join positive prompt parts (e.g., ' ', ', ')"
                    }
                ),
                "negative_joiner": (
                    "STRING", 
                    {
                        "default": DEFAULT_CONFIG["negative_joiner"], 
                        "multiline": False,
                        "tooltip": "String used to join negative prompt parts"
                    }
                ),
                "seed": (
                    "INT", 
                    {
                        "default": DEFAULT_CONFIG["seed"], 
                        "min": 0, 
                        "max": 0x7fffffff, 
                        "step": 1,
                        "tooltip": "Seed for deterministic selection (ignored in random mode)"
                    }
                ),
                "selection_mode": (
                    [mode.value for mode in SelectionMode], 
                    {
                        "default": DEFAULT_CONFIG["selection_mode"],
                        "tooltip": "Strategy for selecting lines from files"
                    }
                ),
            },
            "optional": {
                "reset_counters": (
                    "BOOLEAN", 
                    {
                        "default": False,
                        "tooltip": "Reset incremental counters (useful for incremental mode)"
                    }
                ),
                "file_config_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "JSON config for per-file selection modes. Examples:\n\nCOMPACT (from recovery_config):\n{\"file_configs\":{\"positive/file.txt\":2},\"default_mode\":\"deterministic\"}\n\nEXTENDED:\n{\n  \"file_configs\": {\n    \"positive/styles.txt\": {\"mode\": \"deterministic\", \"seed\": 42},\n    \"positive/subjects.txt\": \"random\",\n    \"negative/avoid.txt\": {\"mode\": \"incremental\", \"start_index\": 0}\n  },\n  \"default_mode\": \"deterministic\",\n  \"default_seed\": 42\n}\nSupported modes: deterministic, incremental, random"
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt", 
        "positive_index",
        "negative_index",
        "debug_info",
        "seed_used",
        "help_info",
        "recovery_config",
    )

    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================

    def assemble(
        self, 
        base_dir: str, 
        positive_joiner: str, 
        negative_joiner: str, 
        seed: int,
        selection_mode: str,
        reset_counters: bool = False,
        file_config_json: str = ""
    ) -> Tuple[str, str, str, str, str, int, str, str]:
        """
        Main assembly function - the entry point called by ComfyUI
        
        Args:
            base_dir: Directory containing positive/ and negative/ subdirs
            positive_joiner: String to join positive prompt parts
            negative_joiner: String to join negative prompt parts  
            seed: Seed for deterministic selection
            selection_mode: Selection strategy ("deterministic", "incremental", "random")
            reset_counters: Whether to reset incremental counters
            
        Returns:
            Tuple of (positive_prompt, negative_prompt, positive_index, 
                     negative_index, debug_info, seed_used, help_info, recovery_config)
        """
        try:
            # Handle counter reset if requested
            if reset_counters:
                self.reset_session()
                self._logger.info("Session counters reset")
            
            # Parse file configuration JSON if provided
            file_config = self._parse_file_config_json(
                file_config_json, 
                default_mode=selection_mode, 
                default_seed=seed
            )
            
            # Validate and analyze directory structure
            analysis = self._analyze_directory_structure(base_dir)
            if not analysis.is_valid:
                error_msg = f"Directory validation failed: {'; '.join(analysis.errors)}"
                return self._create_error_response(error_msg, seed)
            
            # Store analysis for potential future use
            self._last_analysis = analysis
            
            # Perform the actual assembly
            result = self._perform_assembly(
                analysis=analysis,
                positive_joiner=positive_joiner,
                negative_joiner=negative_joiner,
                seed=seed,
                selection_mode=selection_mode,
                file_config=file_config
            )
            
            if not result.success:
                return self._create_error_response(result.error_message or "Assembly failed", seed)
                
            # Format and return results
            return self._format_results(result)
            
        except Exception as e:
            self._logger.error(f"Unexpected error in assemble: {e}")
            return self._create_error_response(f"Unexpected error: {str(e)}", seed)

    def reset_session(self) -> None:
        """Reset the incremental counters for this session"""
        self._session_counters.clear()
        self._logger.info("Incremental session counters cleared")

    def _parse_file_config_json(
        self, 
        json_str: str, 
        default_mode: str = "deterministic", 
        default_seed: int = 42
    ) -> GlobalFileConfig:
        """
        Parse JSON configuration for per-file selection modes
        
        Args:
            json_str: JSON string with file configurations
            default_mode: Default selection mode if not specified
            default_seed: Default seed if not specified
            
        Returns:
            GlobalFileConfig object with parsed configurations
        """
        # Return default config if no JSON provided
        if not json_str or not json_str.strip():
            return GlobalFileConfig(
                default_mode=default_mode,
                default_seed=default_seed
            )
        
        try:
            import json
            config_data = json.loads(json_str)
            
            # Extract global defaults
            global_default_mode = config_data.get("default_mode", default_mode)
            global_default_seed = config_data.get("default_seed", default_seed)
            
            # Parse file-specific configurations
            file_configs = {}
            file_configs_data = config_data.get("file_configs", {})
            
            for file_path, file_config_data in file_configs_data.items():
                if isinstance(file_config_data, str):
                    # Simple format: "file.txt": "mode"
                    file_config = FileConfig(
                        mode=file_config_data,
                        seed=global_default_seed
                    )
                elif isinstance(file_config_data, (int, float)):
                    # Compact recovery format: "file.txt": 2 (index number)
                    file_config = FileConfig(
                        mode="deterministic",
                        seed=int(file_config_data)  # Use the index as seed
                    )
                elif isinstance(file_config_data, dict):
                    # Extended format: "file.txt": {"mode": "...", "seed": ..., "start_index": ...}
                    file_config = FileConfig(
                        mode=file_config_data.get("mode", global_default_mode),
                        seed=file_config_data.get("seed", global_default_seed),
                        start_index=file_config_data.get("start_index", 0)
                    )
                else:
                    self._logger.warning(f"Invalid configuration for file {file_path}: {file_config_data}")
                    continue
                
                file_configs[file_path] = file_config
            
            result = GlobalFileConfig(
                file_configs=file_configs,
                default_mode=global_default_mode,
                default_seed=global_default_seed
            )
            
            self._logger.info(f"Parsed file configuration for {len(file_configs)} files")
            return result
            
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in file_config_json: {e}")
            return GlobalFileConfig(
                default_mode=default_mode,
                default_seed=default_seed
            )
        except Exception as e:
            self._logger.error(f"Error parsing file configuration: {e}")
            return GlobalFileConfig(
                default_mode=default_mode,
                default_seed=default_seed
            )

    def get_directory_info(self, base_dir: str) -> Optional[DirectoryAnalysis]:
        """
        Get detailed information about a directory structure without processing
        
        Args:
            base_dir: Directory to analyze
            
        Returns:
            DirectoryAnalysis object with detailed information, or None if error
        """
        try:
            return self._analyze_directory_structure(base_dir)
        except Exception as e:
            self._logger.error(f"Error analyzing directory {base_dir}: {e}")
            return None

    # =============================================================================
    # DIRECTORY ANALYSIS & VALIDATION
    # =============================================================================

    def _analyze_directory_structure(self, base_dir: str) -> DirectoryAnalysis:
        """
        Comprehensive analysis of directory structure and contents
        
        Args:
            base_dir: Base directory path to analyze
            
        Returns:
            DirectoryAnalysis object with complete information
        """
        base_path = Path(base_dir).expanduser().resolve()
        analysis = DirectoryAnalysis(
            base_path=base_path,
            positive_dir=base_path / "positive",
            negative_dir=base_path / "negative"
        )
        
        # Check base directory existence
        if not base_path.exists():
            analysis.errors.append(f"Base directory does not exist: {base_path}")
            return analysis
            
        if not base_path.is_dir():
            analysis.errors.append(f"Base path is not a directory: {base_path}")
            return analysis
        
        # Check required subdirectories
        for subdir_name in REQUIRED_SUBDIRS:
            subdir_path = base_path / subdir_name
            if not subdir_path.exists():
                analysis.errors.append(f"Missing required subdirectory: {subdir_path}")
            elif not subdir_path.is_dir():
                analysis.errors.append(f"Required path is not a directory: {subdir_path}")
        
        # If basic structure is invalid, return early
        if analysis.errors:
            return analysis
        
        # Analyze positive directory
        analysis.positive_files, pos_line_count = self._analyze_text_files(analysis.positive_dir)
        analysis.total_positive_lines = pos_line_count
        
        # Analyze negative directory  
        analysis.negative_files, neg_line_count = self._analyze_text_files(analysis.negative_dir)
        analysis.total_negative_lines = neg_line_count
        
        # Check for files
        if not analysis.positive_files:
            analysis.errors.append(f"No .txt files found in positive directory: {analysis.positive_dir}")
        if not analysis.negative_files:
            analysis.errors.append(f"No .txt files found in negative directory: {analysis.negative_dir}")
        
        # Generate warnings for empty files
        for file_path in analysis.positive_files + analysis.negative_files:
            try:
                if file_path.stat().st_size == 0:
                    analysis.warnings.append(f"Empty file detected: {file_path.name}")
            except OSError:
                analysis.warnings.append(f"Cannot read file: {file_path.name}")
        
        # Mark as valid if no errors
        analysis.is_valid = len(analysis.errors) == 0
        
        self._logger.info(f"Directory analysis complete: {len(analysis.positive_files)} positive files, "
                         f"{len(analysis.negative_files)} negative files, "
                         f"{len(analysis.errors)} errors, {len(analysis.warnings)} warnings")
        
        return analysis

    def _analyze_text_files(self, directory: Path) -> Tuple[List[Path], int]:
        """
        Analyze text files in a directory
        
        Args:
            directory: Directory to analyze
            
        Returns:
            Tuple of (file_list, total_line_count)
        """
        if not directory.exists():
            return [], 0
            
        txt_files = sorted(directory.glob(DEFAULT_CONFIG["file_extension"]))
        total_lines = 0
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding=DEFAULT_CONFIG["encoding"]) as f:
                    lines = [line.strip() for line in f if line.strip()]
                    total_lines += len(lines)
            except Exception as e:
                self._logger.warning(f"Could not read file {file_path}: {e}")
        
        return txt_files, total_lines

    # =============================================================================
    # PROMPT ASSEMBLY ENGINE
    # =============================================================================

    def _perform_assembly(
        self,
        analysis: DirectoryAnalysis,
        positive_joiner: str,
        negative_joiner: str, 
        seed: int,
        selection_mode: str,
        file_config: GlobalFileConfig
    ) -> AssemblyResult:
        """
        Perform the actual prompt assembly process
        
        Args:
            analysis: Pre-validated directory analysis
            positive_joiner: String to join positive parts
            negative_joiner: String to join negative parts
            seed: Seed for selection
            selection_mode: Selection mode to use
            
        Returns:
            AssemblyResult with all assembly information
        """
        try:
            # Assemble positive prompts
            pos_prompt, pos_indices, pos_debug = self._assemble_from_directory(
                directory=analysis.positive_dir,
                joiner=positive_joiner,
                selection_mode=selection_mode,
                base_seed=seed,
                subdir_name="positive",
                file_config=file_config
            )
            
            # Assemble negative prompts
            neg_prompt, neg_indices, neg_debug = self._assemble_from_directory(
                directory=analysis.negative_dir,
                joiner=negative_joiner,
                selection_mode=selection_mode,
                base_seed=seed,
                subdir_name="negative",
                file_config=file_config
            )
            
            return AssemblyResult(
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                positive_indices=pos_indices,
                negative_indices=neg_indices,
                positive_debug=pos_debug,
                negative_debug=neg_debug,
                seed_used=seed,
                selection_mode=selection_mode,
                file_config_used=file_config,
                success=True
            )
            
        except Exception as e:
            self._logger.error(f"Assembly failed: {e}")
            return AssemblyResult(
                positive_prompt="",
                negative_prompt="",
                positive_indices=[],
                negative_indices=[],
                positive_debug=[],
                negative_debug=[],
                seed_used=seed,
                selection_mode=selection_mode,
                file_config_used=file_config,
                success=False,
                error_message=str(e)
            )

    def _assemble_from_directory(
        self, 
        directory: Path, 
        joiner: str, 
        selection_mode: str,
        base_seed: int,
        subdir_name: str,
        file_config: GlobalFileConfig
    ) -> Tuple[str, List[int], List[PromptSelectionInfo]]:
        """
        Assemble prompt from a single directory with comprehensive tracking
        
        Args:
            directory: Directory containing .txt files
            joiner: String to join selected lines
            selection_mode: Mode for line selection
            base_seed: Base seed for selection
            subdir_name: Name of subdirectory (for logging/debugging)
            
        Returns:
            Tuple of (assembled_prompt, indices_list, debug_info_list)
        """
        txt_files = sorted(directory.glob(DEFAULT_CONFIG["file_extension"]))
        
        if not txt_files:
            self._logger.warning(f"No text files found in {directory}")
            return "", [], []

        parts = []
        indices = []
        debug_info = []

        for file_idx, txt_file in enumerate(txt_files):
            try:
                # Read and clean lines from file
                lines = self._read_file_lines(txt_file)
                
                if not lines:
                    self._logger.debug(f"Skipping empty file: {txt_file.name}")
                    continue
                
                # Get file-specific configuration
                relative_path = f"{subdir_name}/{txt_file.name}"
                file_specific_config = file_config.get_config_for_file(relative_path)
                
                # Determine the selection mode and seed for this specific file
                file_selection_mode = file_specific_config.mode
                file_seed = file_specific_config.seed if file_specific_config.seed is not None else (base_seed + file_idx)
                
                # Select line based on file-specific configuration
                selected_line, selected_idx = self._select_line_from_file(
                    lines=lines,
                    file_key=f"{subdir_name}_{txt_file.name}",
                    selection_mode=file_selection_mode,
                    seed=file_seed,
                    file_config=file_specific_config
                )
                
                # Store results
                parts.append(selected_line)
                indices.append(selected_idx)
                
                # Create debug information with file-specific details
                debug_info.append(PromptSelectionInfo(
                    file_name=txt_file.name,
                    selected_line=selected_line,
                    selected_index=selected_idx,
                    total_lines=len(lines),
                    selection_mode=file_selection_mode,  # Use file-specific mode
                    seed_used=file_seed,
                    file_path=str(txt_file)
                ))
                
            except Exception as e:
                error_msg = f"Error processing file {txt_file}: {e}"
                self._logger.error(error_msg)
                raise ValueError(error_msg)

        assembled_prompt = joiner.join(parts)
        
        self._logger.info(f"Assembled {subdir_name} prompt from {len(parts)} files: "
                         f"'{assembled_prompt[:100]}{'...' if len(assembled_prompt) > 100 else ''}'")
        
        return assembled_prompt, indices, debug_info

    def _read_file_lines(self, file_path: Path) -> List[str]:
        """
        Read and clean lines from a text file
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of non-empty, stripped lines
        """
        try:
            with open(file_path, 'r', encoding=DEFAULT_CONFIG["encoding"]) as f:
                lines = [line.strip() for line in f if line.strip()]
                return lines
        except UnicodeDecodeError:
            # Try alternate encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = [line.strip() for line in f if line.strip()]
                        self._logger.warning(f"File {file_path.name} read with {encoding} encoding")
                        return lines
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Cannot decode file {file_path.name} with any supported encoding")

    def _select_line_from_file(
        self, 
        lines: List[str], 
        file_key: str, 
        selection_mode: str, 
        seed: int,
        file_config: Optional[FileConfig] = None
    ) -> Tuple[str, int]:
        """
        Select a line from file based on the specified selection mode
        
        Args:
            lines: List of non-empty lines from the file
            file_key: Unique identifier for this file (for incremental mode)
            selection_mode: Selection strategy
            seed: Seed for deterministic mode
            
        Returns:
            Tuple of (selected_line, selected_index)
        """
        if not lines:
            return "", 0
        
        if selection_mode == SelectionMode.DETERMINISTIC.value:
            # Predictable, reproducible selection
            idx = seed % len(lines)
            return lines[idx], idx
            
        elif selection_mode == SelectionMode.INCREMENTAL.value:
            # Sequential cycling with session persistence
            if file_key not in self._session_counters:
                # Use start_index from file_config if available
                start_index = 0
                if file_config and file_config.start_index is not None:
                    start_index = file_config.start_index
                self._session_counters[file_key] = start_index
            
            idx = self._session_counters[file_key] % len(lines)
            self._session_counters[file_key] += 1
            return lines[idx], idx
            
        elif selection_mode == SelectionMode.RANDOM.value:
            # True random selection
            idx = random.randint(0, len(lines) - 1)
            return lines[idx], idx
            
        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")

    # =============================================================================
    # RESULT FORMATTING & ERROR HANDLING
    # =============================================================================

    def _format_results(self, result: AssemblyResult) -> Tuple[str, str, str, str, str, int, str, str]:
        """
        Format assembly results for ComfyUI output
        
        Args:
            result: AssemblyResult to format
            
        Returns:
            Formatted tuple for ComfyUI
        """
        # Format indices as comma-separated strings
        pos_indices_str = ",".join(map(str, result.positive_indices))
        neg_indices_str = ",".join(map(str, result.negative_indices))
        
        # Create comprehensive debug information
        debug_info = self._create_debug_info(result)
        
        # Create help information
        help_info = self._create_help_info()
        
        # Create recovery configuration
        recovery_config = self._create_recovery_config(result)
        
        return (
            result.positive_prompt,
            result.negative_prompt,
            pos_indices_str,
            neg_indices_str,
            debug_info,
            result.seed_used,
            help_info,
            recovery_config
        )

    def _create_debug_info(self, result: AssemblyResult) -> str:
        """
        Create comprehensive debug information string with enhanced visual formatting
        
        Args:
            result: AssemblyResult containing debug data
            
        Returns:
            Formatted debug information string with clear visual separation
        """
        debug_lines = [
            "",
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                    PROMPT CONCATENATE DEBUG INFO             ║",
            "╚═══════════════════════════════════════════════════════════════╝",
            "",
            f"🔧 Global Selection Mode: {result.selection_mode.upper()}",
            f"🎲 Base Seed: {result.seed_used}",
            f"⏰ Timestamp: {self._get_timestamp()}",
            ""
        ]
        
        # Show file configuration info if using per-file settings
        if result.file_config_used and result.file_config_used.file_configs:
            debug_lines.extend([
                "┌─────────────────────────────────────────────────────────────┐",
                "│                    📋 FILE CONFIGURATIONS                   │",
                "└─────────────────────────────────────────────────────────────┘",
                "",
                f"  🎯 Default Mode: {result.file_config_used.default_mode}",
                f"  🎲 Default Seed: {result.file_config_used.default_seed}",
                f"  📄 Custom Configs: {len(result.file_config_used.file_configs)} files",
                ""
            ])
            
            for file_path, config in result.file_config_used.file_configs.items():
                mode_info = f"mode={config.mode}"
                if config.seed is not None:
                    mode_info += f", seed={config.seed}"
                if config.start_index is not None:
                    mode_info += f", start_idx={config.start_index}"
                debug_lines.append(f"     📄 {file_path}: {mode_info}")
            
            debug_lines.extend(["", ""])
        else:
            debug_lines.append("")
        
        # Positive files section
        if result.positive_debug:
            debug_lines.extend([
                "┌─────────────────────────────────────────────────────────────┐",
                "│                       ✅ POSITIVE FILES                     │",
                "└─────────────────────────────────────────────────────────────┘",
                ""
            ])
            for idx, info in enumerate(result.positive_debug, 1):
                debug_lines.append(f"  {idx:2d}. 📄 {info}")
            debug_lines.extend(["", ""])
        
        # Negative files section  
        if result.negative_debug:
            debug_lines.extend([
                "┌─────────────────────────────────────────────────────────────┐",
                "│                       ⛔ NEGATIVE FILES                     │",
                "└─────────────────────────────────────────────────────────────┘",
                ""
            ])
            for idx, info in enumerate(result.negative_debug, 1):
                debug_lines.append(f"  {idx:2d}. 📄 {info}")
            debug_lines.extend(["", ""])
        
        # Incremental counters (if applicable)
        if result.selection_mode == SelectionMode.INCREMENTAL.value and self._session_counters:
            debug_lines.extend([
                "┌─────────────────────────────────────────────────────────────┐",
                "│                    🔢 INCREMENTAL COUNTERS                 │",
                "└─────────────────────────────────────────────────────────────┘",
                ""
            ])
            for key, count in sorted(self._session_counters.items()):
                debug_lines.append(f"     🔢 {key}: {count}")
            debug_lines.extend(["", ""])
        
        # Generated Prompts Section
        debug_lines.extend([
            "┌─────────────────────────────────────────────────────────────┐",
            "│                     🎯 GENERATED PROMPTS                    │",
            "└─────────────────────────────────────────────────────────────┘",
            ""
        ])
        
        # Positive prompt
        if result.positive_prompt:
            debug_lines.extend([
                f"  ✅ POSITIVE PROMPT ({len(result.positive_prompt)} characters):",
                "  ┌─────────────────────────────────────────────────────────┐"
            ])
            # Dividir el prompt en líneas que quepan en la caja
            prompt_lines = self._wrap_text_for_box(result.positive_prompt, 55)
            for line in prompt_lines:
                debug_lines.append(f"  │ {line:55} │")
            debug_lines.extend([
                "  └─────────────────────────────────────────────────────────┘",
                ""
            ])
        else:
            debug_lines.extend([
                "  ❌ POSITIVE PROMPT: (empty)",
                ""
            ])
        
        # Negative prompt
        if result.negative_prompt:
            debug_lines.extend([
                f"  ⛔ NEGATIVE PROMPT ({len(result.negative_prompt)} characters):",
                "  ┌─────────────────────────────────────────────────────────┐"
            ])
            # Dividir el prompt en líneas que quepan en la caja
            prompt_lines = self._wrap_text_for_box(result.negative_prompt, 55)
            for line in prompt_lines:
                debug_lines.append(f"  │ {line:55} │")
            debug_lines.extend([
                "  └─────────────────────────────────────────────────────────┘",
                ""
            ])
        else:
            debug_lines.extend([
                "  ❌ NEGATIVE PROMPT: (empty)",
                ""
            ])

        # Summary
        debug_lines.extend([
            "┌─────────────────────────────────────────────────────────────┐",
            "│                        📊 SUMMARY                           │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
            f"  📁 Files processed: {len(result.positive_debug + result.negative_debug)}",
            f"  📊 Total characters: {len(result.positive_prompt) + len(result.negative_prompt)}",
            f"  🎯 Positive parts: {len(result.positive_indices)}",
            f"  🚫 Negative parts: {len(result.negative_indices)}",
            "",
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                         END DEBUG INFO                       ║",
            "╚═══════════════════════════════════════════════════════════════╝",
            ""
        ])
        
        return "\n".join(debug_lines)

    def _create_error_response(self, error_message: str, seed: int) -> Tuple[str, str, str, str, str, int, str, str]:
        """
        Create a standardized error response
        
        Args:
            error_message: Error message to include
            seed: Seed value to return
            
        Returns:
            Error response tuple for ComfyUI
        """
        error_debug = f"""
═══════════════════════════════════════
    PROMPT CONCATENATE ERROR
═══════════════════════════════════════
❌ ERROR: {error_message}

🔧 TROUBLESHOOTING:
  1. Check that base directory exists
  2. Ensure 'positive' and 'negative' subdirectories exist
  3. Verify .txt files are present in both subdirectories
  4. Check file permissions and encoding
  5. Review file contents for proper formatting

💡 DIRECTORY STRUCTURE EXAMPLE:
  txt/
  ├── positive/
  │   ├── styles.txt
  │   ├── subjects.txt
  │   └── moods.txt
  └── negative/
      ├── avoid.txt
      └── problems.txt

Timestamp: {self._get_timestamp()}
═══════════════════════════════════════
"""
        
        # Create help information even for errors
        help_info = self._create_help_info()
        
        # Create empty recovery config for errors
        error_recovery = '{"error": "No recovery possible due to error", "timestamp": "' + self._get_timestamp() + '"}'
        
        return ("", "", "", "", error_debug.strip(), seed, help_info, error_recovery)

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _create_help_info(self) -> str:
        """
        Create helpful information about selection modes and usage
        
        Returns:
            Formatted help information string
        """
        help_text = """
╔══════════════════════════════════════════════════════════════════╗
║                    📚 PROMPT CONCATENATE HELP                    ║
╚══════════════════════════════════════════════════════════════════╝

🎯 SELECTION MODES:

┌──────────────────────────────────────────────────────────────────┐
│ 🎯 DETERMINISTIC (Reproducible)                                 │
└──────────────────────────────────────────────────────────────────┘
• Uses formula: seed % number_of_lines  
• Same seed = same result ALWAYS
• Perfect for consistency and reproducible results
• Example: seed=42 with 5 lines → always selects line 2

┌──────────────────────────────────────────────────────────────────┐
│ 🔄 INCREMENTAL (Sequential)                                     │
└──────────────────────────────────────────────────────────────────┘
• Goes line by line: 0, 1, 2, 3... then back to 0
• Maintains counter per file across session
• Perfect for systematic exploration of all options
• Example: 1st run=line 0, 2nd run=line 1, 3rd run=line 2...

┌──────────────────────────────────────────────────────────────────┐
│ 🎲 RANDOM (Maximum Variety)                                     │
└──────────────────────────────────────────────────────────────────┘
• Completely random selection each time
• Different result every execution
• Perfect for maximum unpredictability and variety
• Example: Could select any line randomly each time

📋 JSON CONFIGURATION EXAMPLES:

💡 COMPACT (from recovery_config):
{"file_configs":{"positive/file.txt":2},"default_mode":"deterministic"}

📝 EXTENDED (manual configuration):
{
  "file_configs": {
    "positive/styles.txt": "deterministic",
    "positive/subjects.txt": "random", 
    "negative/problems.txt": "incremental"
  },
  "default_mode": "deterministic",
  "default_seed": 42
}

💡 BEST PRACTICES:
• Use DETERMINISTIC for consistent base elements
• Use INCREMENTAL to cycle through variations systematically  
• Use RANDOM for maximum creative variety
• Combine modes for optimal balance of control and surprise

📁 DIRECTORY STRUCTURE REQUIRED:
base_dir/
├── positive/    (files for positive prompt parts)
└── negative/    (files for negative prompt parts)

🔄 RECOVERY SYSTEM:
• The 'recovery_config' output contains a compact JSON to reproduce results
• Copy the entire recovery_config and paste it as file_config_json input
• Ultra-compact format: {"file_configs":{"positive/file.txt":2},"default_mode":"deterministic"}
• Perfect for saving favorite combinations and reproducing specific results

╔══════════════════════════════════════════════════════════════════╗
║                           EQX - 2024                            ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return help_text.strip()

    def _create_recovery_config(self, result: AssemblyResult) -> str:
        """
        Create a compact recovery configuration JSON that can reproduce this exact result
        
        Args:
            result: AssemblyResult containing the current execution data
            
        Returns:
            Compact JSON string with essential recovery configuration
        """
        import json
        
        # Create compact file configurations using short format
        file_configs = {}
        
        # Process all files with minimal data
        for info in result.positive_debug:
            file_configs[f"positive/{info.file_name}"] = info.selected_index
        
        for info in result.negative_debug:
            file_configs[f"negative/{info.file_name}"] = info.selected_index
        
        # Build ultra-compact recovery configuration
        recovery_config = {
            "file_configs": file_configs,
            "default_mode": "deterministic"
        }
        
        # Add minimal metadata
        recovery_config["_meta"] = {
            "timestamp": self._get_timestamp(),
            "seed": result.seed_used,
            "files": len(file_configs)
        }
        
        # Add incremental counters only if present and different from default
        if hasattr(self, '_session_counters') and self._session_counters:
            recovery_config["_counters"] = dict(self._session_counters)
        
        try:
            # Return ultra-compact JSON (no indentation for maximum efficiency)
            return json.dumps(recovery_config, separators=(',', ':'), ensure_ascii=False)
        except Exception as e:
            # Minimal fallback
            return f'{{"error": "Recovery failed: {str(e)}", "timestamp": "{self._get_timestamp()}"}}'

    def _wrap_text_for_box(self, text: str, max_width: int) -> List[str]:
        """
        Wrap text to fit within a specified width, preserving words when possible
        
        Args:
            text: Text to wrap
            max_width: Maximum characters per line
            
        Returns:
            List of wrapped lines that fit within max_width
        """
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Si la palabra sola es más larga que max_width, la dividimos
            if len(word) > max_width:
                # Si hay contenido en la línea actual, la agregamos primero
                if current_line:
                    lines.append(current_line.strip())
                    current_line = ""
                
                # Dividir la palabra larga en chunks
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                
                # El resto de la palabra va a la línea actual
                if word:
                    current_line = word + " "
            else:
                # Verificar si la palabra cabe en la línea actual
                test_line = current_line + word
                if len(test_line) <= max_width:
                    current_line = test_line + " "
                else:
                    # La palabra no cabe, guardar la línea actual y empezar nueva
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
        
        # Agregar la última línea si tiene contenido
        if current_line.strip():
            lines.append(current_line.strip())
        
        # Si no hay líneas, retornar al menos una línea vacía
        if not lines:
            lines = [""]
            
        return lines


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PromptConcatenateUnified", 
    "PromptSelectionInfo", 
    "DirectoryAnalysis",
    "FileConfig",
    "GlobalFileConfig",
    "AssemblyResult"
]

# Node metadata for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Prompt Concatenate Unified - EQX": PromptConcatenateUnified
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Concatenate Unified - EQX": "Prompt Concatenate Unified - EQX"
} 