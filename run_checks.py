import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
from tqdm import tqdm

# --- Dependency Imports ---
# It is required to install: nudenet, facexlib, tqdm
# pip install nudenet facexlib torch numpy opencv-python Pillow tqdm

try:
    from nudenet import NudeDetector
except ImportError:
    tqdm.write("Warning: nudenet library not found. Please run 'pip install nudenet'. NSFW check will not be available.")
    NudeDetector = None

try:
    from facexlib.detection import init_detection_model, RetinaFace
except ImportError:
    tqdm.write("Warning: facexlib library not found. Please run 'pip install facexlib'. Face detection checks will not be available.")
    RetinaFace = None
    def init_detection_model(model_name, model_rootpath): return None


# --- NSFW Categories and Constants ---

ALL_CATEGORIES = [
    'EXPOSED_ANUS', 'BUTTOCKS_EXPOSED', 'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F',
    'BELLY_EXPOSED', 'FEMALE_BREAST_COVERED', 'FEMALE_GENITALIA_COVERED', 
    'BUTTOCKS_COVERED', 'COVERED_FEET', 'EXPOSED_FEET', 'ARMPITS_EXPOSED', 
    'FACE_FEMALE', 'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'OTHER', 'NOT_DETECTED'
]

CATEGORY_DISPLAY_MAPPINGS = {
    'EXPOSED_ANUS': 'Exposed Anus',
    'BUTTOCKS_EXPOSED': 'Buttocks Exposed',
    'EXPOSED_BREAST_F': 'Exposed Breast Female',
    'EXPOSED_GENITALIA_F': 'Exposed Genitalia Female',
    'BELLY_EXPOSED': 'Belly Exposed',
    'FEMALE_BREAST_COVERED': 'Female Breast Covered',
    'FEMALE_GENITALIA_COVERED': 'Female Genitalia Covered',
    'BUTTOCKS_COVERED': 'Buttocks Covered',
    'COVERED_FEET': 'Covered Feet',
    'EXPOSED_FEET': 'Exposed Feet',
    'ARMPITS_EXPOSED': 'Armpits Exposed',
    'FACE_FEMALE': 'Face',
    'FEMALE_BREAST_EXPOSED': 'Female Breast Exposed',
    'FEMALE_GENITALIA_EXPOSED': 'Female Genitalia Exposed',
    'OTHER': 'Other',
    'NOT_DETECTED': 'Not Detected'
}

DEFAULT_THRESHOLDS = {
    'EXPOSED_ANUS': 0.99, 'BUTTOCKS_EXPOSED': 0.40, 'EXPOSED_BREAST_F': 0.35,
    'EXPOSED_GENITALIA_F': 0.35, 'BELLY_EXPOSED': 0.99,
    'FEMALE_BREAST_COVERED': 0.99, 'FEMALE_GENITALIA_COVERED': 0.99,
    'BUTTOCKS_COVERED': 0.40, 'COVERED_FEET': 0.01,
    'EXPOSED_FEET': 0.99, 'ARMPITS_EXPOSED': 0.99, 'FACE_FEMALE': 0.6,
    'FEMALE_BREAST_EXPOSED': 0.35, 'FEMALE_GENITALIA_EXPOSED': 0.33, 'OTHER': 0.6,
    'NOT_DETECTED': 1.0
}

# Default categories considered "NSFW" for a simple True/False check
DEFAULT_NSFW_CATEGORIES = {
    "EXPOSED_GENITALIA_F", "EXPOSED_BREAST_F", "EXPOSED_ANUS",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "BUTTOCKS_COVERED" 
}


# --- Utility Functions ---

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a PyTorch tensor (1, H, W, C) to a PIL Image."""
    return Image.fromarray((image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)).convert("RGB")

def tensor2cv(image: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor (H, W, C) from RGB [0,1] to an OpenCV image (BGR, uint8)."""
    if image.dim() == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    image_np = image.cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch tensor (1, H, W, C)."""
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)

def numpy2tensor(image_np: np.ndarray) -> torch.Tensor:
    """Converts a NumPy array (H, W, C) to a PyTorch tensor (1, H, W, C)."""
    return torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)

def is_video_file(path):
    """Check if the file is a video based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(path.lower().endswith(ext) for ext in video_extensions)


# --- Core Checker Class ---

class ImageChecker:
    """
    A class to perform a series of checks on an image, externalizing logic
    from a ComfyUI workflow.
    """
    def __init__(self, retinaface_model_path=None):
        self.nsfw_detector = None
        self.face_detector = None

        if RetinaFace and init_detection_model:
            tqdm.write("Initializing face detection model...")
            # In the original node, the path is constructed. Here you might need to
            # ensure the model is downloaded or provide a direct path.
            # Default path from node: os.path.join(models_dir, 'facexlib')
            self.face_detector = init_detection_model(
                "retinaface_resnet50", 
                model_rootpath=retinaface_model_path
            )
            if self.face_detector:
                tqdm.write("Face detection model loaded successfully.")
            else:
                tqdm.write("Warning: Could not load face detection model.")
        
        if NudeDetector:
            tqdm.write("Initializing NSFW detector...")
            self.nsfw_detector = NudeDetector()
            tqdm.write("NSFW detector loaded successfully.")

    def check_nsfw(self, image_tensor: torch.Tensor, allowed_categories: set = None) -> (bool, str):
        """
        Checks if an image is NSFW based on a set of allowed categories.
        Returns a tuple: (is_nsfw: bool, best_category: str)
        """
        if not self.nsfw_detector:
            tqdm.write("NSFW detector not available.")
            return False, "N/A"
            
        if allowed_categories is None:
            allowed_categories = DEFAULT_NSFW_CATEGORIES

        img_pil = tensor2pil(image_tensor)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            img_pil.save(tmp.name)
            try:
                results = self.nsfw_detector.detect(tmp.name)
                best_category = "NOT_DETECTED"
                max_score = -1.0
                
                if results:
                    for detection in results:
                        label = detection.get('class') or detection.get('label')
                        score = detection.get('score', 0)
                        
                        # We check all detections, but only consider the ones in allowed_categories for the final "best_category"
                        if score > max_score and label in allowed_categories and score >= DEFAULT_THRESHOLDS.get(label, 0.99):
                            max_score = score
                            best_category = label
                
                is_nsfw = best_category != "NOT_DETECTED"
                return is_nsfw, best_category

            except Exception as e:
                tqdm.write(f"Error during NSFW detection: {e}")
                return False, "ERROR"

    def check_face_count(self, image_tensor: torch.Tensor, confidence: float = 0.8) -> int:
        """
        Counts the number of faces in an image.
        Returns the face count as an integer.
        """
        if not self.face_detector:
            tqdm.write("Face detector not available.")
            return -1
            
        with torch.no_grad():
            # The model expects a BGR numpy array
            cv_image = tensor2cv(image_tensor)
            bboxes = self.face_detector.detect_faces(cv_image, confidence)
        
        return len(bboxes) if bboxes is not None else 0

    def check_partial_face(self, image_tensor: torch.Tensor, confidence: float = 0.8, margin: float = 0.5) -> bool:
        """
        Checks if the primary detected face is partially outside the image frame after a margin is applied.
        Returns a boolean: True if the face is partial, False otherwise.
        """
        if not self.face_detector:
            tqdm.write("Face detector not available.")
            return False

        with torch.no_grad():
            cv_image = tensor2cv(image_tensor)
            bboxes_with_landmarks = self.face_detector.detect_faces(cv_image, confidence)

        if bboxes_with_landmarks is None or len(bboxes_with_landmarks) == 0:
            return False # No faces detected, so no partial face

        img_height, img_width, _ = cv_image.shape
        
        # Use the first detected face (highest confidence)
        x0, y0, x1, y1, *_ = bboxes_with_landmarks[0]
        bbox = (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))

        # Logic from is_bbox_partially_outside in FaceDetectOut
        x, y, w, h = bbox
        margin_w_half = int(w * margin / 2)
        margin_h_half = int(h * margin / 2)

        is_out = (
            (x - margin_w_half) < 0 or
            (y - margin_h_half) < 0 or
            (x + w + margin_w_half) > img_width or
            (y + h + margin_h_half) > img_height
        )
        return is_out

def generate_filename(original_path, results):
    """Generates a new filename for 'flat' mode based on check results."""
    base, ext = os.path.splitext(os.path.basename(original_path))
    parts = [base]

    # Format: (nombre de la imagen)_(yes_nsfw)_(faces:number)_(face (IN-OUT) frame)
    
    if 'is_nsfw' in results:
        parts.append("yes_nsfw" if results['is_nsfw'] else "no_nsfw")
    
    if 'face_count' in results:
        parts.append(f"faces_{results['face_count']}")
    
    if 'is_partial_face' in results:
        status = "OUT" if results['is_partial_face'] else "IN"
        parts.append(f"face_{status}_frame")
    
    new_name = "_".join(parts) + ext
    # Sanitize filename
    return "".join(c for c in new_name if c.isalnum() or c in ('_','-','.'))

def get_hierarchical_path(output_dir, results, sort_order):
    """
    Determines the subfolder path for hierarchical sorting based on a dynamic sort_order list.
    sort_order contains keys like 'nsfw', 'faces', 'framing'.
    """
    path_parts = [output_dir]

    for criterion in sort_order:
        if criterion == 'nsfw' and 'is_nsfw' in results:
            path_parts.append("NSFW" if results['is_nsfw'] else "SFW")
        
        elif criterion == 'faces' and 'face_count' in results:
            count = results['face_count']
            if count == 0:
                path_parts.append("0_faces")
            elif count == 1:
                path_parts.append("1_face")
            else:
                path_parts.append("multiple_faces")
        
        elif criterion == 'framing' and 'face_count' in results and results['face_count'] > 0 and 'is_partial_face' in results:
            path_parts.append("partial_face" if results['is_partial_face'] else "complete_face")

    return os.path.join(*path_parts)

def print_custom_logic(results, image_path):
    """Prints suggested actions based on check results."""
    tqdm.write("\n--- SUGGESTED ACTIONS ---")
    if results.get('is_nsfw'):
        tqdm.write(f"Action: Image '{os.path.basename(image_path)}' is NSFW. Consider moving to a review folder.")
    if results.get('face_count', -1) == 0:
        tqdm.write(f"Action: No faces found in '{os.path.basename(image_path)}'. Consider skipping face processing.")
    if results.get('face_count', -1) > 1:
        tqdm.write(f"Action: Multiple faces detected in '{os.path.basename(image_path)}'. Consider flagging for manual check.")
    if results.get('is_partial_face'):
        tqdm.write(f"Action: Face in '{os.path.basename(image_path)}' is partial or close to the edge. Consider re-cropping.")
    if not any(v for k, v in results.items() if k != 'nsfw_category'):
        tqdm.write("No specific actions suggested based on results.")
    tqdm.write("-------------------------\n")

def process_and_report(input_path, checker, args, output_dir=None):
    """
    Processes a single image or video frame, runs checks, prints results,
    and saves the output if an output directory is provided.
    """
    image_pil = None
    input_display_name = os.path.basename(input_path)

    try:
        if is_video_file(input_path):
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                tqdm.write(f"Error: Could not open video file: {input_path}")
                return
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_index = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                tqdm.write(f"Error: Could not read the middle frame from video: {input_path}")
                return
                
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            input_display_name += f" (frame {middle_frame_index})"
        else:
            image_pil = Image.open(input_path).convert("RGB")
    except Exception as e:
        tqdm.write(f"Error loading or processing {input_path}: {e}")
        return

    if image_pil is None:
        tqdm.write(f"Error: Could not get a valid image from {input_path}")
        return
        
    image_tensor = pil2tensor(image_pil)
    results = {}
    
    # Run checks based on arguments
    if 'nsfw' in args.check:
        # For CLI, use default NSFW categories. The GUI will pass its own set.
        is_nsfw, category = checker.check_nsfw(image_tensor)
        results['is_nsfw'] = is_nsfw
        results['nsfw_category'] = category
    
    if 'face_count' in args.check:
        count = checker.check_face_count(image_tensor, confidence=args.confidence)
        results['face_count'] = count

    if 'partial_face' in args.check:
        is_partial = checker.check_partial_face(image_tensor, confidence=args.confidence, margin=args.margin)
        results['is_partial_face'] = is_partial
        
    # Display results
    tqdm.write(f"\n--- RESULTS FOR: {input_display_name} ---")
    for key, value in results.items():
        tqdm.write(f"{key}: {value}")
    tqdm.write("--------------------")

    if output_dir:
        try:
            # The 'flat' vs 'hierarchical' logic is now determined by the sort_order from args
            if args.sort_order:
                target_dir = get_hierarchical_path(output_dir, results, args.sort_order)
                os.makedirs(target_dir, exist_ok=True)
                # In hierarchical mode, we use the original filename
                save_path = os.path.join(target_dir, os.path.basename(input_path))
            else:  # 'flat' mode (no sort order)
                new_filename = generate_filename(input_path, results)
                save_path = os.path.join(output_dir, new_filename)

            image_pil.save(save_path)
            tqdm.write(f"Saved processed image to: {save_path}")
        except Exception as e:
            tqdm.write(f"Error saving file for {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Perform checks on an image, video, or directory of media.",
        epilog="Example: python run_checks.py C:\\path\\to\\images --output-dir C:\\output --save-mode hierarchical"
    )
    parser.add_argument("input_path", type=str, help="Path to the input image, video, or directory.")
    parser.add_argument("--check", nargs='+', choices=['nsfw', 'face_count', 'partial_face'], default=['nsfw', 'face_count', 'partial_face'], help="Specify which checks to run.")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold for face detection.")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for partial face check.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save processed images.")
    parser.add_argument("--sort-order", nargs='*', choices=['nsfw', 'faces', 'framing'], default=[], help="Defines the hierarchical sorting order. If empty, saves in a flat structure.")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        tqdm.write(f"Error: Input path does not exist: {args.input_path}")
        return

    if args.output_dir and not os.path.isdir(args.output_dir):
        tqdm.write(f"Output directory not found. Creating it at: {args.output_dir}")
        os.makedirs(args.output_dir)

    checker = ImageChecker(retinaface_model_path="Nodes")

    if os.path.isdir(args.input_path):
        tqdm.write(f"Processing all items in directory: {args.input_path}")
        
        # Filter for files to avoid iterating over subdirectories
        file_list = [item for item in sorted(os.listdir(args.input_path)) if os.path.isfile(os.path.join(args.input_path, item))]
        
        with tqdm(total=len(file_list), desc="Processing media") as pbar:
            for item in file_list:
                item_path = os.path.join(args.input_path, item)
                process_and_report(item_path, checker, args, output_dir=args.output_dir)
                pbar.update(1)
                pbar.set_postfix_str(os.path.basename(item_path), refresh=True)

    else:
        tqdm.write(f"Processing single file: {args.input_path}")
        process_and_report(args.input_path, checker, args, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 