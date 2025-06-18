import gradio as gr
import numpy as np
import sys
import os
from PIL import Image
import cv2
import time
from tqdm import tqdm
import concurrent.futures

# Import the core logic from the main script
try:
    from run_checks import (
        ImageChecker, numpy2tensor, pil2tensor, is_video_file,
        CATEGORY_DISPLAY_MAPPINGS, DEFAULT_NSFW_CATEGORIES, 
        generate_filename, get_hierarchical_path
    )
except ImportError:
    print("Error: Could not import from run_checks.py. Make sure both files are in the same directory.")
    # Provide dummy classes/functions if run_checks.py is missing
    class ImageChecker:
        def __init__(self, *args, **kwargs): print("Dummy ImageChecker initialized.")
        def check_nsfw(self, *args, **kwargs): return False, "UNAVAILABLE"
        def check_face_count(self, *args, **kwargs): return -1
        def check_partial_face(self, *args, **kwargs): return False
    def numpy2tensor(image_np): return None
    def pil2tensor(image_pil): return None
    def is_video_file(path): return False
    CATEGORY_DISPLAY_MAPPINGS = {}
    DEFAULT_NSFW_CATEGORIES = set()
    def generate_filename(path, results): return "dummy_filename.png"
    def get_hierarchical_path(path, results, order): return "dummy/path"


def launch_gradio_ui():
    """Launches the Gradio web interface."""

    def get_action_text(results):
        # Generates a human-readable summary of actions based on check results
        actions = []
        if results.get('is_nsfw'): actions.append("Image is NSFW. Suggest moving to a review folder.")
        if results.get('face_count', -1) == 0: actions.append("No faces found. Suggest skipping face-related processing.")
        if results.get('face_count', -1) > 1: actions.append("Multiple faces detected. Suggest flagging for manual check.")
        if results.get('is_partial_face'): actions.append("Face is partial or too close to the edge. Suggest re-cropping or rejection.")
        return "\n".join(actions) if actions else "No specific actions suggested."

    def gradio_fn(mode, image_np, dir_path, output_dir, save_mode, sort_level1, sort_level2, sort_level3, do_nsfw, do_face_count, do_partial_face, confidence, margin, batch_size, *nsfw_categories):
        # --- 1. Validation & Setup ---
        if mode == "Single Image" and image_np is None: return {}, "Please upload an image first."
        if mode == "Directory" and not os.path.isdir(dir_path): return {}, f"Error: Directory not found at '{dir_path}'"
        if not output_dir: return {}, "Error: An output directory must be specified."
        
        if not os.path.isdir(output_dir):
            try: os.makedirs(output_dir)
            except Exception as e: return {}, f"Error creating output directory: {e}"

        # --- 2. Determine Sort Order ---
        sort_order = []
        if "Jerárquico (por defecto)" in save_mode:
            sort_order = ['nsfw', 'faces', 'framing']
        elif "Jerárquico (personalizado)" in save_mode:
            sort_map = {"Estado NSFW": "nsfw", "Conteo de Caras": "faces", "Encuadre de Cara": "framing"}
            sort_order = [sort_map[level] for level in [sort_level1, sort_level2, sort_level3] if level != "Ninguno"]

        # --- 3. Initialization ---
        tqdm.write("Initializing models...")
        checker = ImageChecker(retinaface_model_path="Nodes")
        tqdm.write("Models initialized.")

        # --- 4. Nested Logic Functions ---
        def run_checks_on_tensor(image_tensor):
            results = {}
            if do_nsfw:
                cat_keys = [k for k in sorted(CATEGORY_DISPLAY_MAPPINGS.keys()) if k != 'NOT_DETECTED']
                allowed_categories = {cat_keys[i] for i, is_checked in enumerate(nsfw_categories) if is_checked}
                is_nsfw, category = checker.check_nsfw(image_tensor, allowed_categories=allowed_categories)
                results.update({'is_nsfw': is_nsfw, 'nsfw_category': category})
            if do_face_count:
                results['face_count'] = checker.check_face_count(image_tensor, confidence=confidence)
            if do_partial_face and results.get('face_count', 0) > 0:
                 results['is_partial_face'] = checker.check_partial_face(image_tensor, confidence=confidence, margin=margin)
            return results

        def save_image(image_to_save, original_filename, results, output_dir, current_sort_order):
            try:
                if current_sort_order:  # Hierarchical mode
                    target_dir = get_hierarchical_path(output_dir, results, current_sort_order)
                    os.makedirs(target_dir, exist_ok=True)
                    save_path = os.path.join(target_dir, os.path.basename(original_filename))
                else:  # Flat mode
                    new_filename = generate_filename(original_filename, results)
                    save_path = os.path.join(output_dir, new_filename)
                image_to_save.save(save_path)
                return f"Saved to {save_path}"
            except Exception as e:
                return f"FAILED to save {original_filename}: {e}"

        # --- 5. Processing ---
        if mode == "Single Image":
            results = run_checks_on_tensor(numpy2tensor(image_np))
            action_text = get_action_text(results)
            original_filename = f"uploaded_image_{int(time.time())}.png"
            save_log = save_image(Image.fromarray(image_np), original_filename, results, output_dir, sort_order)
            return results, action_text + f"\n\n{save_log}"
        
        elif mode == "Directory":
            image_files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            num_files = len(image_files)
            log = f"Found {num_files} files. Starting processing with {batch_size} workers...\n"
            yield {}, log

            report_stats = {"processed": 0, "failed_to_load": 0, "nsfw": {"SFW": 0, "NSFW": 0}, "faces": {"0": 0, "1": 0, "multiple": 0}, "framing": {"complete": 0, "partial": 0}}

            def process_image_task(filename):
                """Task to be run in a thread. Loads, checks, and saves one image."""
                original_path = os.path.join(dir_path, filename)
                image_pil = None
                try:
                    if is_video_file(original_path):
                        cap = cv2.VideoCapture(original_path)
                        if cap.isOpened():
                            # Correctly read a frame from the middle of the video
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if frame_count > 0:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        image_pil = Image.open(original_path).convert("RGB")
                except Exception as e:
                    return filename, None, f"FAILED to load: {e}"
                
                if image_pil:
                    results = run_checks_on_tensor(pil2tensor(image_pil))
                    save_log = save_image(image_pil, filename, results, output_dir, sort_order)
                    return filename, results, save_log
                else:
                    return filename, None, "SKIPPED (not valid image/video)"

            with tqdm(total=num_files, desc="Processing Directory") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Create a dictionary to map futures to their original filenames
                    futures = {executor.submit(process_image_task, filename): filename for filename in image_files}
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        filename, results, task_log = future.result()
                        log_prefix = f"({i+1}/{num_files}) {filename}: "
                        log += log_prefix + task_log + "\n"
                        
                        # Update statistics based on the results from the thread
                        if results:
                            report_stats["processed"] += 1
                            if 'is_nsfw' in results:
                                report_stats["nsfw"]["NSFW" if results['is_nsfw'] else "SFW"] += 1
                            if 'face_count' in results:
                                count = results['face_count']
                                if count == 0:
                                    report_stats["faces"]["0"] += 1
                                elif count == 1:
                                    report_stats["faces"]["1"] += 1
                                else:
                                    report_stats["faces"]["multiple"] += 1
                                if count > 0 and 'is_partial_face' in results:
                                    report_stats["framing"]["partial" if results['is_partial_face'] else "complete"] += 1
                        else:
                            report_stats["failed_to_load"] += 1

                        pbar.set_postfix_str(filename, refresh=True)
                        pbar.update(1)
                        yield {}, log
            
            summary = "\n\n--- Processing Report ---\n"
            summary += f"Total files: {num_files}, Processed: {report_stats['processed']}, Failed: {report_stats['failed_to_load']}\n"
            if do_nsfw: summary += f"NSFW Distribution: SFW ({report_stats['nsfw']['SFW']}), NSFW ({report_stats['nsfw']['NSFW']})\n"
            if do_face_count: summary += f"Face Distribution: None ({report_stats['faces']['0']}), Single ({report_stats['faces']['1']}), Multiple ({report_stats['faces']['multiple']})\n"
            if do_partial_face: summary += f"Framing Distribution: Complete ({report_stats['framing']['complete']}), Partial ({report_stats['framing']['partial']})\n"
            yield {}, log + summary

    # --- 6. UI Layout ---
    nsfw_checkboxes = []
    with gr.Blocks() as iface:
        gr.Markdown("# Image Verifier\nUpload an image or specify a directory and select the checks to perform.")
        with gr.Row():
            with gr.Column(scale=1):
                # Input & Output
                input_mode = gr.Radio(["Single Image", "Directory"], label="Input Mode", value="Single Image")
                image_input = gr.Image(label="Upload Image", type="numpy", visible=True)
                dir_input = gr.Textbox(label="Input Directory Path", placeholder="e.g., C:\\path\\to\\images", visible=False, value=os.path.join(os.getcwd(), "images", "source"))
                output_dir_input = gr.Textbox(label="Output Directory Path", placeholder="e.g., C:\\path\\to\\output", value=os.path.join(os.getcwd(), "images", "output"))
                
                batch_size_slider = gr.Slider(1, 4, 1, step=1, label="Batch Processes (for Directory mode)", visible=False)

                input_mode.change(
                    lambda mode: (gr.update(visible=mode=="Single Image"), gr.update(visible=mode=="Directory"), gr.update(visible=mode=="Directory")), 
                    input_mode, 
                    [image_input, dir_input, batch_size_slider]
                )
                
                # Sorting Configuration
                save_mode_input = gr.Radio(
                    ["Plano (nombres largos)", "Jerárquico (por defecto)", "Jerárquico (personalizado)"], 
                    label="Save Mode", 
                    value="Plano (nombres largos)"
                )
                
                with gr.Group(visible=False) as custom_sort_group:
                    sort_choices = ["Ninguno", "Estado NSFW", "Conteo de Caras", "Encuadre de Cara"]
                    sort_level1 = gr.Dropdown(sort_choices, label="Nivel 1 de Clasificación", value="Estado NSFW")
                    sort_level2 = gr.Dropdown(sort_choices, label="Nivel 2 de Clasificación", value="Conteo de Caras")
                    sort_level3 = gr.Dropdown(sort_choices, label="Nivel 3 de Clasificación", value="Encuadre de Cara")

                save_mode_input.change(
                    fn=lambda mode: gr.update(visible="personalizado" in mode),
                    inputs=save_mode_input,
                    outputs=custom_sort_group
                )

                # Check Configuration
                with gr.Accordion("Check Configuration", open=False):
                    do_nsfw_checkbox = gr.Checkbox(label="Check for NSFW", value=True)
                    with gr.Accordion("Select NSFW Categories to Detect", open=False) as nsfw_category_group:
                        for internal_name, display_name in sorted(CATEGORY_DISPLAY_MAPPINGS.items()):
                            if internal_name != 'NOT_DETECTED': nsfw_checkboxes.append(gr.Checkbox(label=display_name, value=(internal_name in DEFAULT_NSFW_CATEGORIES)))
                    do_nsfw_checkbox.change(lambda x: gr.update(visible=x), do_nsfw_checkbox, nsfw_category_group)
                    do_face_count_checkbox = gr.Checkbox(label="Check Face Count", value=True)
                    do_partial_face_checkbox = gr.Checkbox(label="Check for Partial Face", value=True)
                    confidence_slider = gr.Slider(0.1, 1.0, 0.8, step=0.05, label="Face Detection Confidence")
                    margin_slider = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Partial Face Margin")

                submit_btn = gr.Button("Run Checks")

            with gr.Column(scale=1):
                json_output = gr.JSON(label="Check Results")
                action_output = gr.Textbox(label="Actions & Log", lines=15)
        
        all_inputs = [input_mode, image_input, dir_input, output_dir_input, save_mode_input, sort_level1, sort_level2, sort_level3, do_nsfw_checkbox, do_face_count_checkbox, do_partial_face_checkbox, confidence_slider, margin_slider, batch_size_slider] + nsfw_checkboxes
        submit_btn.click(fn=gradio_fn, inputs=all_inputs, outputs=[json_output, action_output])

    print("Launching Gradio interface... Go to http://127.0.0.1:7860")
    iface.launch()

if __name__ == "__main__":
    launch_gradio_ui() 