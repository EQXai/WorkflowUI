"""
Modular UI sections for WorkflowUI application.
This module contains organized UI section builders to break down the monolithic
launch_gui function into manageable, reusable components.
"""

from pathlib import Path
from typing import List, Tuple, Any, Dict

import gradio as gr
from app_config import CONFIG
from gui_utils import (
    create_number_input, create_textbox_input, create_log_textbox,
    get_default_node_value, load_workflow_data, list_presets,
    create_visibility_toggle_callback
)
from run_checks import CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES
import orchestrator as oc

# =============================================================================
# PRESET CONFIGURATION SECTION
# =============================================================================

def create_preset_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the configuration presets section with search, save, load functionality.
    
    Returns:
        Tuple of Gradio components for preset management
    """
    with gr.Accordion("Configuration Presets", open=False):
        gr.Markdown("")

        with gr.Row():
            # Left column: list & filter
            with gr.Column(scale=1):
                filter_txt = gr.Textbox(
                    label="Search presets", 
                    placeholder="Type to filter by name", 
                    interactive=True,
                )

                from gui_utils import list_presets
                presets_dd = gr.Radio(
                    label="Available presets",
                    choices=list_presets(),
                    interactive=True,
                    info="Select a preset to load, rename, delete or export",
                )

            # Right column: create / rename / import
            with gr.Column(scale=1):
                preset_name_txt = gr.Textbox(
                    label="Preset name",
                    placeholder="MyNewPreset",
                    info="Create a new preset or overwrite an existing one",
                )
                save_preset_btn = gr.Button("Save / Overwrite")
                
                with gr.Row():
                    load_preset_btn = gr.Button("Load")
                    delete_preset_btn = gr.Button("Delete")
                    export_btn = gr.Button("Export")
                    reset_btn = gr.Button("Reset defaults")

                rename_txt = gr.Textbox(label="Rename to", placeholder="New name")
                rename_btn = gr.Button("Rename")
                gr.Markdown("**Import preset (.json)**")
                import_file = gr.File(
                    label="Import file",
                    file_types=[".json"],
                    interactive=True,
                )

    return (
        filter_txt, presets_dd, preset_name_txt, save_preset_btn,
        load_preset_btn, delete_preset_btn, export_btn, reset_btn,
        rename_txt, rename_btn, import_file
    )

def create_main_controls_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the main pipeline control section with batch size, run/cancel buttons.
    
    Returns:
        Tuple of main control components
    """
    with gr.Row():
        with gr.Column(scale=3, min_width=120):  # Batch Size
            batch_runs_input = gr.Number(
                value=1, minimum=1, precision=0,
                label="Batch Size",
                info="Images to generate"
            )
        with gr.Column(scale=14, min_width=400):  # Run Pipeline & Cancel
            run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
            cancel_btn = gr.Button("⏹️ Cancel", variant="stop")
        with gr.Column(scale=3, min_width=120):  # Endless Mode & Sound Alert & Workflow2 Selection
            endless_until_cancel_cb = gr.Checkbox(label="🔁 Endless Mode", value=False)
            play_sound_cb = gr.Checkbox(label="🔊 Sound Alert", value=True)
            workflow2_mode_cb = gr.Checkbox(label="⚡ One Step Mode", value=False, 
                                           info="Use OneStep (checked) or TwoSteps (unchecked)")

    return batch_runs_input, run_btn, cancel_btn, endless_until_cancel_cb, play_sound_cb, workflow2_mode_cb

def create_status_boxes_section():
    """Create the status display boxes."""
    with gr.Row():
        pending_box = gr.Markdown("Pending: 0", elem_id="pending-box", elem_classes=["pending-box"])
        status_box = gr.Markdown("Idle", elem_id="status-box", elem_classes=["pending-box"])
        metrics_box = gr.Markdown("Approved: 0 | Rejected: 0", elem_id="metrics-box", elem_classes=["pending-box"])

    return pending_box, status_box, metrics_box

def create_advanced_checks_section():
    """Create the advanced checks configuration section."""
    with gr.Accordion("🔧 Advanced Checks", open=False):
        # Main Checks
        do_nsfw_cb = gr.Checkbox(label="🔞 NSFW Detection", value=True)
        do_face_count_cb = gr.Checkbox(label="👤 Face Count", value=True)
        do_partial_face_cb = gr.Checkbox(label="👁️ Partial Face Detection", value=True)
        
        with gr.Accordion("NSFW Categories", open=False) as nsfw_cat_group:
            nsfw_cat_checkboxes = []
            for internal, display in sorted(CATEGORY_DISPLAY_MAPPINGS.items()):
                if internal == "NOT_DETECTED":
                    continue
                # Desactivar por defecto BUTTOCKS_COVERED y BUTTOCKS_EXPOSED
                default_value = (internal in DEFAULT_NSFW_CATEGORIES) and internal not in ["BUTTOCKS_COVERED", "BUTTOCKS_EXPOSED"]
                nsfw_cat_checkboxes.append(
                    gr.Checkbox(label=display, value=default_value)
                )

        # Hide/show category selection depending on NSFW toggle
        do_nsfw_cb.change(lambda x: gr.update(visible=x), do_nsfw_cb, nsfw_cat_group)

        with gr.Accordion("Stop Criteria", open=False):
            stop_multi_faces_cb = gr.Checkbox(
                label="Stop if multiple faces", value=True
            )
            stop_partial_face_cb = gr.Checkbox(
                label="Stop if partial face", value=True
            )

        confidence_slider = gr.Slider(
            CONFIG.validation.MIN_CONFIDENCE, 
            CONFIG.validation.MAX_CONFIDENCE, 
            CONFIG.validation.DEFAULT_CONFIDENCE, 
            step=0.05,
            label="Detection Confidence",
            info="Higher = stricter (0.1-1.0)"
        )

        margin_slider = gr.Slider(
            CONFIG.validation.MIN_MARGIN, 
            CONFIG.validation.MAX_MARGIN, 
            CONFIG.validation.DEFAULT_MARGIN, 
            step=0.1,
            label="Partial Face Margin",
            info="Extra margin tolerance"
        )

    return (
        do_nsfw_cb, do_face_count_cb, do_partial_face_cb, nsfw_cat_checkboxes,
        nsfw_cat_group, stop_multi_faces_cb, stop_partial_face_cb,
        confidence_slider, margin_slider
    )

def create_seed_settings_section():
    """Create the seed configuration section."""
    with gr.Accordion("⚙️ Seed Settings", open=False):
        seed_mode_radio = gr.Radio(
            choices=["Incremental", "Random", "Static"],
            value="Random",
            label="Seed Mode",
        )
        seed_counter_input = gr.Number(
            value=0, minimum=0, maximum=CONFIG.seed.MAX_SEED_VALUE, precision=0,
            label="Seed Value",
            info="Starting/static seed value"
        )

    return seed_mode_radio, seed_counter_input

def create_output_prompts_section():
    """Create the output configuration and prompts section."""
    with gr.Accordion("💾 Output & Prompts", open=False):
        # Custom filename
        override_filename_cb = gr.Checkbox(label="Custom filename", value=False)
        filename_text = gr.Textbox(label="Filename base", value="final_image")

        # Characteristics text
        characteristics_text = gr.Textbox(
            label="Character Description",
            lines=2,
            value=CONFIG.prompts.DEFAULT_CHARACTERISTICS,
        )

        # Direct Prompt Loader
        load_prompts_cb = gr.Checkbox(label="Manual prompt list", value=False)
        prompts_textbox = gr.Textbox(
            label="Prompt list",
            lines=3,
            placeholder="{{{title}}}{{positive}}{negative}, …",
            visible=False,
        )

        # Toggle visibility of the prompt textbox
        load_prompts_cb.change(
            lambda x: gr.update(visible=x), 
            inputs=[load_prompts_cb], 
            outputs=[prompts_textbox]
        )

    return (
        override_filename_cb, filename_text, characteristics_text,
        load_prompts_cb, prompts_textbox
    )

def create_images_gallery_section():
    """Create the images display and gallery section."""
    # Workflow 1 Image
    wf1_img_out = gr.Image(
        label="Workflow 1 Image",
        interactive=False,
        height=CONFIG.ui.PREVIEW_IMAGE_HEIGHT,
        elem_classes=["preview-img"],
    )

    # Results Gallery
    try:
        from gradio.components import Carousel  # Gradio >=4
        album_gallery = Carousel(
            label=f"Results (last {CONFIG.ui.MAX_GALLERY_IMAGES})",
            visible=True,
            scale=2,
        )
    except ImportError:
        album_gallery = gr.Gallery(
            label=f"Results (last {CONFIG.ui.MAX_GALLERY_IMAGES})",
            columns=CONFIG.ui.GALLERY_COLUMNS,
            preview=False,
            elem_id="results-gallery",
            scale=2,
        )

    sound_audio = gr.Audio(label="Sound", autoplay=True, visible=False)

    return wf1_img_out, album_gallery, sound_audio

def create_logs_section():
    """Create the logs display section."""
    with gr.Accordion("Log", open=True):
        log_text = create_log_textbox(lines=CONFIG.ui.DEFAULT_LOG_LINES)
    
    with gr.Accordion("Seeds Log", open=False):
        seeds_log_text = create_log_textbox(lines=CONFIG.ui.DEFAULT_SEED_LOG_LINES)
    
    with gr.Accordion("Prompts Log", open=False):
        prompt_log_text = create_log_textbox(lines=CONFIG.ui.DEFAULT_PROMPT_LOG_LINES)

    with gr.Accordion("Output Filename", open=False):
        filename_log_text = create_log_textbox(lines=2)

    return log_text, seeds_log_text, prompt_log_text, filename_log_text

def create_directory_section():
    """Create directory input section."""
    output_dir_input = gr.Textbox(
        label="📂 ComfyUI Output", 
        value=str(oc.DEFAULT_OUTPUT_DIR)
    )
    final_dir_input = gr.Textbox(
        label="📁 Final Images", 
        value=str((Path.home()/"ComfyUI"/"final").expanduser())
    )
    
    return output_dir_input, final_dir_input



# =============================================================================
# STATUS BOXES SECTION
# =============================================================================

def create_status_boxes_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the status display boxes for pending, status, and metrics.
    
    Returns:
        Tuple of status display components
    """
    with gr.Row():
        pending_box = gr.Markdown("Pending: 0", elem_id="pending-box", elem_classes=["pending-box"])
        status_box = gr.Markdown("Idle", elem_id="status-box", elem_classes=["pending-box"])
        metrics_box = gr.Markdown("Approved: 0 | Rejected: 0", elem_id="metrics-box", elem_classes=["pending-box"])

    return pending_box, status_box, metrics_box

# =============================================================================
# DIRECTORY INPUTS SECTION
# =============================================================================

def create_directory_section() -> Tuple[gr.components.Component, ...]:
    """
    Create directory input section for output and final directories.
    
    Returns:
        Tuple of directory input components
    """
    output_dir_input = gr.Textbox(
        label="📂 ComfyUI Output", 
        value=str(oc.DEFAULT_OUTPUT_DIR)
    )
    final_dir_input = gr.Textbox(
        label="📁 Final Images", 
        value=str((Path.home()/"ComfyUI"/"final").expanduser())
    )
    
    return output_dir_input, final_dir_input

# =============================================================================
# ADVANCED CHECKS SECTION
# =============================================================================

def create_advanced_checks_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the advanced checks configuration section.
    
    Returns:
        Tuple of check configuration components
    """
    with gr.Accordion("🔧 Advanced Checks", open=False):
        # Main Checks
        do_nsfw_cb = gr.Checkbox(label="🔞 NSFW Detection", value=True)
        do_face_count_cb = gr.Checkbox(label="👤 Face Count", value=True)
        do_partial_face_cb = gr.Checkbox(label="👁️ Partial Face Detection", value=True)
        
        with gr.Accordion("NSFW Categories", open=False) as nsfw_cat_group:
            nsfw_cat_checkboxes = []
            for internal, display in sorted(CATEGORY_DISPLAY_MAPPINGS.items()):
                if internal == "NOT_DETECTED":
                    continue
                nsfw_cat_checkboxes.append(
                    gr.Checkbox(label=display, value=(internal in DEFAULT_NSFW_CATEGORIES))
                )

        # Hide/show category selection depending on NSFW toggle
        do_nsfw_cb.change(lambda x: gr.update(visible=x), do_nsfw_cb, nsfw_cat_group)

        with gr.Accordion("Stop Criteria", open=False):
            stop_multi_faces_cb = gr.Checkbox(
                label="Stop if multiple faces", value=False
            )
            stop_partial_face_cb = gr.Checkbox(
                label="Stop if partial face", value=False
            )

        confidence_slider = gr.Slider(
            CONFIG.validation.MIN_CONFIDENCE, 
            CONFIG.validation.MAX_CONFIDENCE, 
            CONFIG.validation.DEFAULT_CONFIDENCE, 
            step=0.05,
            label="Detection Confidence",
            info="Higher = stricter (0.1-1.0)"
        )

        margin_slider = gr.Slider(
            CONFIG.validation.MIN_MARGIN, 
            CONFIG.validation.MAX_MARGIN, 
            CONFIG.validation.DEFAULT_MARGIN, 
            step=0.1,
            label="Partial Face Margin",
            info="Extra margin tolerance"
        )

    return (
        do_nsfw_cb, do_face_count_cb, do_partial_face_cb, nsfw_cat_checkboxes,
        nsfw_cat_group, stop_multi_faces_cb, stop_partial_face_cb,
        confidence_slider, margin_slider
    )

# =============================================================================
# SEED SETTINGS SECTION
# =============================================================================

def create_seed_settings_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the seed configuration section.
    
    Returns:
        Tuple of seed setting components
    """
    with gr.Accordion("⚙️ Seed Settings", open=False):
        seed_mode_radio = gr.Radio(
            choices=["Incremental", "Random", "Static"],
            value="Incremental",
            label="Seed Mode",
        )
        seed_counter_input = gr.Number(
            value=0, minimum=0, maximum=CONFIG.seed.MAX_SEED_VALUE, precision=0,
            label="Seed Value",
            info="Starting/static seed value"
        )

    return seed_mode_radio, seed_counter_input

# =============================================================================
# OUTPUT & PROMPTS SECTION
# =============================================================================

def create_output_prompts_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the output configuration and prompts section.
    
    Returns:
        Tuple of output and prompt components
    """
    with gr.Accordion("💾 Output & Prompts", open=False):
        # Custom filename
        override_filename_cb = gr.Checkbox(label="Custom filename", value=False)
        filename_text = gr.Textbox(label="Filename base", value="final_image")

        # Characteristics text
        characteristics_text = gr.Textbox(
            label="Character Description",
            lines=2,
            value=CONFIG.prompts.DEFAULT_CHARACTERISTICS,
        )

        # Direct Prompt Loader
        load_prompts_cb = gr.Checkbox(label="Manual prompt list", value=False)
        prompts_textbox = gr.Textbox(
            label="Prompt list",
            lines=3,
            placeholder="{{{title}}}{{positive}}{negative}, …",
            visible=False,
        )

        # Toggle visibility of the prompt textbox
        load_prompts_cb.change(
            lambda x: gr.update(visible=x), 
            inputs=[load_prompts_cb], 
            outputs=[prompts_textbox]
        )

    return (
        override_filename_cb, filename_text, characteristics_text,
        load_prompts_cb, prompts_textbox
    )

# =============================================================================
# IMAGES AND GALLERY SECTION
# =============================================================================

def create_images_gallery_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the images display and gallery section.
    
    Returns:
        Tuple of image and gallery components
    """
    # Workflow 1 Image (double height and square)
    wf1_img_out = gr.Image(
        label="Workflow 1 Image",
        interactive=False,
        height=CONFIG.ui.PREVIEW_IMAGE_HEIGHT * 2,  # Double the height (800px)
        width=CONFIG.ui.PREVIEW_IMAGE_HEIGHT * 2,   # Make it square (800x800px)
        elem_classes=["preview-img"],
    )

    # Results Gallery
    try:
        from gradio.components import Carousel  # Gradio >=4
        album_gallery = Carousel(
            label=f"Results (last {CONFIG.ui.MAX_GALLERY_IMAGES})",
            visible=True,
            scale=2,
        )
    except ImportError:
        album_gallery = gr.Gallery(
            label=f"Results (last {CONFIG.ui.MAX_GALLERY_IMAGES})",
            columns=CONFIG.ui.GALLERY_COLUMNS,
            preview=False,
            elem_id="results-gallery",
            scale=2,
        )

    sound_audio = gr.Audio(label="Sound", autoplay=True, visible=False)

    return wf1_img_out, album_gallery, sound_audio

# =============================================================================
# WORKFLOW COMPONENT FACTORY
# =============================================================================

def create_workflow_component(
    node_id: str, 
    param: str, 
    label: str, 
    node_data: Dict, 
    param_type: type,
    default_value: Any = None
) -> gr.components.Component:
    """
    Factory function to create workflow components consistently.
    
    Args:
        node_id: Workflow node ID
        param: Parameter name within the node
        label: Display label for the component
        node_data: Node data dictionary
        param_type: Type of the parameter (int, float, str)
        default_value: Default value if not found in node
        
    Returns:
        Appropriate Gradio component
    """
    current_value = get_default_node_value(
        node_data.get(node_id, {}), 
        param, 
        default_value or (CONFIG.workflow.DEFAULT_STEPS if param == "steps" else 0)
    )
    
    if param_type == int:
        return create_number_input(label, current_value, precision=0)
    elif param_type == float:
        return create_number_input(label, current_value, precision=1)
    else:  # str
        return create_textbox_input(label, str(current_value))

# =============================================================================
# WORKFLOW SETTINGS SECTION
# =============================================================================

def create_workflow_settings_section() -> Tuple[List[gr.components.Component], List[gr.components.Component], List, List]:
    """
    Create the advanced workflow settings section with both workflows.
    
    Returns:
        Tuple of (override_components, override_components2, WF1_MAPPING, WF2_MAPPING)
    """
    with gr.Accordion("🔧 Advanced Workflow Settings", open=False):
        override_components = []
        override_components2 = []
        WF1_OVERRIDE_MAPPING = []
        WF2_OVERRIDE_MAPPING = []

        # Load workflow data
        wf1_data = load_workflow_data(oc.WORKFLOW1_JSON)
        wf2_data = load_workflow_data(oc.WORKFLOW2_JSON)

        with gr.Row():
            # Left column: Workflow 1
            with gr.Column(scale=1):
                wf1_mode_radio = gr.Radio(
                    choices=["Normal", "PromptConcatenate"],
                    value="Normal",
                    label="Workflow 1 Mode",
                )

                gr.Markdown("### Workflow 1")

                # Define WF1 components configuration
                wf1_configs = [
                    ("168", "steps", "Steps KSampler 1", int, CONFIG.workflow.DEFAULT_STEPS),
                    ("169", "steps", "Steps KSampler 2", int, CONFIG.workflow.DEFAULT_STEPS),
                    ("170", "text", "Prefix Title", str, ""),
                    ("173", "ckpt_name", "Checkpoint", str, ""),
                    ("176", "width", "Width", int, CONFIG.workflow.DEFAULT_WIDTH),
                    ("176", "height", "Height", int, CONFIG.workflow.DEFAULT_HEIGHT),
                    ("180", "text", "Text Node", str, ""),
                    ("182", "steps", "Steps KSampler 3", int, CONFIG.workflow.DEFAULT_STEPS),
                    ("190", "file_path", "File Path - 190 (relative to texts/)", str, "stand1.txt"),
                ]

                for node_id, param, label, param_type, default_val in wf1_configs:
                    if node_id == "176" and param == "height":
                        # Skip height, it was handled with width in a Row
                        continue
                    elif node_id == "176" and param == "width":
                        # Handle width and height together
                        gr.Markdown("**Resolution**")
                        with gr.Row():
                            comp_w = create_workflow_component(node_id, "width", "Width", wf1_data, int, CONFIG.workflow.DEFAULT_WIDTH)
                            comp_h = create_workflow_component(node_id, "height", "Height", wf1_data, int, CONFIG.workflow.DEFAULT_HEIGHT)
                        override_components.extend([comp_w, comp_h])
                        WF1_OVERRIDE_MAPPING.extend([
                            ("176", "width", int),
                            ("176", "height", int),
                        ])
                    else:
                        comp = create_workflow_component(node_id, param, label, wf1_data, param_type, default_val)
                        override_components.append(comp)
                        WF1_OVERRIDE_MAPPING.append((node_id, param, param_type))

            # Right column: Workflow 2
            with gr.Column(scale=1):
                gr.Markdown("### Workflow 2")

                # Define WF2 components configuration
                wf2_configs = [
                    ("248", "text", "Text – 248", str, ""),
                    ("224", "guidance", "Guidance", float, CONFIG.workflow.DEFAULT_GUIDANCE),
                    ("226", "grain_power", "Grain power", float, CONFIG.workflow.DEFAULT_GRAIN_POWER),
                    ("230", "size", "Max Size", int, CONFIG.workflow.DEFAULT_MAX_SIZE),
                    ("239", "steps", "Steps – 239", int, CONFIG.workflow.DEFAULT_STEPS),
                    ("239", "denoise", "Denoise – 239", float, CONFIG.workflow.DEFAULT_DENOISE),
                    ("242", "prompt", "CR Prompt Test – 242", str, ""),
                    ("243", "prompt", "CR Prompt Test – 243", str, ""),
                    ("287", "guidance", "Guidance – 287", float, CONFIG.workflow.DEFAULT_GUIDANCE),
                    ("289", "grain_power", "Grain power – 289", float, CONFIG.workflow.DEFAULT_GRAIN_POWER),
                    ("293", "size", "Max size - 293", int, CONFIG.workflow.DEFAULT_MAX_SIZE),
                    ("302", "steps", "Steps – 302", int, CONFIG.workflow.DEFAULT_STEPS),
                    ("302", "denoise", "Denoise – 302", float, CONFIG.workflow.DEFAULT_DENOISE),
                    ("305", "prompt", "Prompt – 305", str, ""),
                    ("306", "prompt", "Prompt – 306", str, ""),
                ]

                for node_id, param, label, param_type, default_val in wf2_configs:
                    if node_id == "239" and param == "denoise":
                        # Skip denoise, it was handled with steps
                        continue
                    elif node_id == "302" and param == "denoise":
                        # Skip denoise, it was handled with steps
                        continue
                    elif node_id == "239" and param == "steps":
                        # Handle steps and denoise together for node 239
                        comp_steps = create_workflow_component(node_id, "steps", "Steps – 239", wf2_data, int, CONFIG.workflow.DEFAULT_STEPS)
                        comp_denoise = create_workflow_component(node_id, "denoise", "Denoise – 239", wf2_data, float, CONFIG.workflow.DEFAULT_DENOISE)
                        override_components2.extend([comp_steps, comp_denoise])
                        WF2_OVERRIDE_MAPPING.extend([
                            ("239", "steps", int),
                            ("239", "denoise", float),
                        ])
                    elif node_id == "302" and param == "steps":
                        # Handle steps and denoise together for node 302
                        comp_steps = create_workflow_component(node_id, "steps", "Steps – 302", wf2_data, int, CONFIG.workflow.DEFAULT_STEPS)
                        comp_denoise = create_workflow_component(node_id, "denoise", "Denoise – 302", wf2_data, float, CONFIG.workflow.DEFAULT_DENOISE)
                        override_components2.extend([comp_steps, comp_denoise])
                        WF2_OVERRIDE_MAPPING.extend([
                            ("302", "steps", int),
                            ("302", "denoise", float),
                        ])
                    else:
                        comp = create_workflow_component(node_id, param, label, wf2_data, param_type, default_val)
                        override_components2.append(comp)
                        WF2_OVERRIDE_MAPPING.append((node_id, param, param_type))

        # Return mode selector as first component
        all_override_components = [wf1_mode_radio] + override_components
        
    return all_override_components, override_components2, WF1_OVERRIDE_MAPPING, WF2_OVERRIDE_MAPPING

# =============================================================================
# LORA CONFIGURATION SECTION
# =============================================================================

def create_lora_section() -> Tuple[gr.components.Component, ...]:
    """
    Create the LoRA configuration section with auto-fill and manual controls.
    
    Returns:
        Tuple of LoRA configuration components
    """
    with gr.Accordion("🎨 LoRA Configuration", open=False):
        gr.Markdown("*Leave blank to use workflow defaults*")
        
        with gr.Accordion("Auto-fill from directory", open=False):
            auto_lora_cb = gr.Checkbox(label="Enable auto-fill", value=False)
            lora_dir_tb = gr.Textbox(label="LoRA directory", placeholder="/path/to/lora/folder")
            populate_btn = gr.Button("Auto Select")
        
        with gr.Row():
            # Node 244
            with gr.Column(scale=1):
                gr.Markdown("### Node 244")
                lora244_inputs = []
                lora244_strengths = []
                for idx in range(1, CONFIG.workflow.MAX_LORA_SLOTS + 1):
                    with gr.Row():
                        txt = gr.Textbox(label=f"LoRA {idx}", placeholder="KimberlyMc1.safetensors")
                        default_strength = (CONFIG.workflow.DEFAULT_LORA_STRENGTH_PRIMARY 
                                          if idx == 1 else CONFIG.workflow.DEFAULT_LORA_STRENGTH_SECONDARY)
                        sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="str")
                    lora244_inputs.append(txt)
                    lora244_strengths.append(sl)

            # Node 307
            with gr.Column(scale=1):
                gr.Markdown("### Node 307")
                lora307_inputs = []
                lora307_strengths = []
                for idx in range(1, CONFIG.workflow.MAX_LORA_SLOTS + 1):
                    with gr.Row():
                        txt = gr.Textbox(label=f"LoRA {idx}", placeholder="KimberlyMc1.safetensors")
                        default_strength = (CONFIG.workflow.DEFAULT_LORA_STRENGTH_PRIMARY 
                                          if idx == 1 else CONFIG.workflow.DEFAULT_LORA_STRENGTH_SECONDARY)
                        sl = gr.Slider(0.0, 1.5, default_strength, step=0.05, label="str")
                    lora307_inputs.append(txt)
                    lora307_strengths.append(sl)

    return (
        auto_lora_cb, lora_dir_tb, populate_btn,
        lora244_inputs, lora244_strengths,
        lora307_inputs, lora307_strengths
    )

# =============================================================================
# MAIN LAYOUT BUILDER
# =============================================================================

def create_main_layout() -> Tuple[gr.components.Component, ...]:
    """
    Create the main three-column layout with images, controls, and logs.
    
    Returns:
        Tuple of main layout components
    """
    with gr.Row():
        # Left column: Controls and settings
        with gr.Column(scale=1):
            # Directory inputs
            output_dir_input, final_dir_input = create_directory_section()
            
            # Advanced checks
            checks_components = create_advanced_checks_section()
            
            # Seed settings  
            seed_mode_radio, seed_counter_input = create_seed_settings_section()
            
            # Output & prompts
            output_components = create_output_prompts_section()

        # Middle column: Images and Gallery
        with gr.Column(scale=2):
            wf1_img_out, album_gallery, sound_audio = create_images_gallery_section()

        # Right column: Logs
        with gr.Column(scale=1):
            log_text, seeds_log_text, prompt_log_text, filename_log_text = create_logs_section()

    # Combine all components for easier return
    all_components = (
        output_dir_input, final_dir_input,
        *checks_components,
        seed_mode_radio, seed_counter_input,
        *output_components,
        wf1_img_out, album_gallery, sound_audio,
        log_text, seeds_log_text, prompt_log_text, filename_log_text
    )
    
    return all_components 