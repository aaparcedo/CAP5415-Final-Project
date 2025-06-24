import argparse
import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import re
from llava.eval.LLaVA_G_Eval import Evaluator_MM_Inter
from llava.constants import DEFAULT_IMAGE_TOKEN

# --------------------------
# FIXED CONFIG
# --------------------------
MODEL_PATH = "/home/da530038/llava-grouding/LLaVA-Grounding/checkpoints/llava_grounding"
VISION_CFG = "configs/openseed/openseed_swint_lang_joint_2st_visual_prompt.yaml"
INTER_CFG = "configs/semsam/visual_prompt_encoder.yaml"
TEMP_IMAGE_PATH = "./temp_llava_frame.jpg"

# --------------------------
# UTILITY FUNCTIONS
# --------------------------

def format_phrases_with_boxes(text, boxes, width, height):
    """
    Match phrases with boxes by index. If mismatched, keep all boxes.
    Unmatched boxes are labeled as 'unmatched_box_{i}'.
    """
    pattern = re.compile(r"<g_s>(.*?)<g_e>")
    phrases = pattern.findall(text)
    phrase_box_dict = {}

    total_phrases = len(phrases)
    total_boxes = len(boxes)
    min_len = min(total_phrases, total_boxes)

    if total_phrases != total_boxes:
        print(f"⚠️ Mismatch: {total_phrases} phrases vs {total_boxes} boxes")
        print(f"Text: {text}")

    # Match common indices
    for i in range(min_len):
        phrase = phrases[i].strip()
        box = boxes[i]
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        abs_box = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        phrase_box_dict[phrase] = abs_box

    # Handle extra boxes
    for i in range(min_len, total_boxes):
        box = boxes[i]
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        abs_box = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        phrase_box_dict[f"unmatched_box_{i}"] = abs_box

    return phrase_box_dict


def unresize_boxes(box_list, img_width, img_height):
    if not box_list:
        return []
    box_tensor = torch.cat(box_list, dim=0).clone()
    ratio = min(img_width, img_height) / max(img_width, img_height)
    if img_width > img_height:
        box_tensor[:, 1] /= ratio
        box_tensor[:, 3] /= ratio
    elif img_width < img_height:
        box_tensor[:, 0] /= ratio
        box_tensor[:, 2] /= ratio
    return box_tensor

# --------------------------
# MODEL WRAPPER
# --------------------------

class LlavaSingleSample:
    def __init__(self, model_path, vision_cfg, inter_cfg, device="cuda"):
        self.model_backend = Evaluator_MM_Inter(
            model_path=model_path,
            path_vision_model_cfg=vision_cfg,
            path_inter_model_cfg=inter_cfg
        )
        self.model_backend.model.to(device)
        self.device = device

    def run_inference(self, image_path: str, question: str):
        input_data = {
            "file_name": image_path,
            "image_id": 0,
            "question_id": 0,
            "conversations": [[[{"from": "human", "value": DEFAULT_IMAGE_TOKEN + " " + question},
                                {"from": "gpt", "value": "Placeholder."}], None]],
            "points": None,
            "mode_inter": None,
            "matching_threshold": 0.3,
            "temporature": 0
        }
        processed = self.model_backend.data_mapper(input_data)[0]
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.to(self.device)
        with torch.no_grad():
            output = self.model_backend.evaluate_sample([processed])
        if len(output) == 4:
            text, boxes, _, _ = output
        else:
            text, boxes, _ = output
        return text, boxes

# --------------------------
# DATA LOADER
# --------------------------

class HCSTVGDataloader:
    def __init__(self, annotation_path, video_dir):
        self.video_dir = video_dir
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = os.path.join(self.video_dir, entry['video_path'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        tube_start = entry.get('tube_start_frame', 0)
        tube_end = entry.get('tube_end_frame', 0)
        frames, frame_count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > tube_end:
                break
            if tube_start <= frame_count <= tube_end:
                frames.append(frame)
            frame_count += 1
        cap.release()
        return frames, entry

# --------------------------
# EVALUATION
# --------------------------

def evaluate_entry_llava(frames, entry, runner, frame_step=15):
    caption = entry.get("caption", "").strip()
    if not caption.endswith("(with grounding)"):
        caption += " (with grounding)"
    orig_W, orig_H = entry.get("width"), entry.get("height")
    if orig_W is None or orig_H is None:
        orig_H, orig_W = frames[0].shape[:2] if frames else (800, 800)
    phrase_boxes_per_frame = {}
    for i, frame in enumerate(frames[::frame_step]):
        frame_H, frame_W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_frame).save(TEMP_IMAGE_PATH)
        text, boxes = runner.run_inference(TEMP_IMAGE_PATH, caption)
        if boxes and len(boxes) > 0:
            corrected_boxes = unresize_boxes(boxes, frame_W, frame_H).cpu().numpy()
            phrase_box_dict = format_phrases_with_boxes(text, corrected_boxes, frame_W, frame_H)
            phrase_boxes_per_frame[str(i)] = phrase_box_dict
        else:
            phrase_boxes_per_frame[str(i)] = {}
    return {
        "video_path": entry.get("video_path", ""),
        "caption": caption,
        "tube_start_frame": entry.get("tube_start_frame", 0),
        "tube_end_frame": entry.get("tube_end_frame", 0),
        "phrase_boxes_per_frame": phrase_boxes_per_frame
    }

# --------------------------
# MAIN
# --------------------------

def main(args):
    runner = LlavaSingleSample(MODEL_PATH, VISION_CFG, INTER_CFG, device=args.device)
    dataset = HCSTVGDataloader(args.anno_path, args.video_dir)
    predictions = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        result = evaluate_entry_llava(frames, entry, runner, frame_step=15)
        predictions.append(result)
    else:
        for idx, (frames, entry) in enumerate(tqdm(dataset, desc="Running LLaVA HC-STVG")):
            result = evaluate_entry_llava(frames, entry, runner, frame_step=15)
            predictions.append(result)
            if args.max_iters > 0 and idx + 1 >= args.max_iters:
                break

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"✅ Done! Results saved to {args.output_path}")

# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Grounded Phrase Evaluation on HC-STVG")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to HC-STVG annotation JSON")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing HC-STVG videos")
    parser.add_argument("--output_path", type=str, default="./results/test_phrase_bbox.json",
                        help="Path to save the predictions JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference (default: cuda)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (>= 0) or -1 for full dataset evaluation")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Maximum number of iterations (batches) to process (use a positive value for testing, -1 for full dataset)")
    args = parser.parse_args()
    main(args)
