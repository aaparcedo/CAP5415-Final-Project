import argparse
import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from llava.eval.LLaVA_G_Eval import Evaluator_MM_Inter
from llava.constants import DEFAULT_IMAGE_TOKEN

# --------------------------
# CONFIGURATION
# --------------------------
MODEL_PATH = "/home/da530038/llava-grouding/LLaVA-Grounding/checkpoints/llava_grounding"
VISION_CFG = "configs/openseed/openseed_swint_lang_joint_2st_visual_prompt.yaml"
INTER_CFG = "configs/semsam/visual_prompt_encoder.yaml"
TEMP_IMAGE_PATH = "./temp_llava_frame_hcstvg1.jpg"

# --------------------------
# LLaVA Inference Wrapper
# --------------------------


class LlavaSingleSample:
    def __init__(self, model_path, vision_cfg, inter_cfg):
        self.model_backend = Evaluator_MM_Inter(
            model_path=model_path,
            path_vision_model_cfg=vision_cfg,
            path_inter_model_cfg=inter_cfg
        )

    def run_inference(self, image_path: str, question: str):
        """
        Run inference using LLaVA via temporary image file.
        Returns text and boxes (normalized [x1, y1, x2, y2]).
        """
        input_data = {
            "file_name": image_path,
            "image_id": 0,
            "question_id": 0,
            "conversations": [[[
                {"from": "human", "value": DEFAULT_IMAGE_TOKEN + " " + question},
                {"from": "gpt", "value": "Placeholder."}
            ], None]],
            "points": None,
            "mode_inter": None,
            "matching_threshold": 0.6,
            "temporature": 0
        }
        processed = self.model_backend.data_mapper(input_data)[0]
        device = self.model_backend.model.device
        for k, v in processed.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.to(device)
        with torch.no_grad():
            output = self.model_backend.evaluate_sample([processed])
        if len(output) == 4:
            text, boxes, masks, matched_masks = output
        else:
            text, boxes, masks = output
        if boxes is None or len(boxes) == 0:
            print("⚠️ No boxes detected. Using default box.")
            boxes = [torch.tensor([[0.1, 0.1, 0.9, 0.9]])]
        return text, boxes

# --------------------------
# IoU Calculation Function
# --------------------------


def calculate_iou(box1, box2):
    """
    Computes IoU for boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2 = x1 + w1, y1 + h1

    x1_gt, y1_gt, w2, h2 = box2
    x2_gt, y2_gt = x1_gt + w2, y1_gt + h2

    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --------------------------
# Aspect Ratio Correction for Boxes
# --------------------------


def unresize_boxes(box_list, img_width, img_height):
    """
    Adjust normalized boxes (x1,y1,x2,y2) for aspect ratio padding.
    """
    if not box_list:
        return []
    # Stack boxes and remove redundant singleton dimension.
    box_tensor = torch.stack(box_list).squeeze(1).clone()  # shape: [N, 4]
    ratio = min(img_width, img_height) / max(img_width, img_height)
    if img_width > img_height:
        box_tensor[:, 1] /= ratio
        box_tensor[:, 3] /= ratio
    elif img_width < img_height:
        box_tensor[:, 0] /= ratio
        box_tensor[:, 2] /= ratio
    return box_tensor

# --------------------------
# HC-STVG Data Loader (for LLaVA)
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

        # Use tube_start_frame and tube_end_frame from entry
        tube_start = entry.get('tube_start_frame', 0)
        tube_end = entry.get('tube_end_frame', 0)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > tube_end:
                break
            if tube_start <= frame_count <= tube_end:
                frames.append(frame)  # keep original frame (BGR)
            frame_count += 1
        cap.release()
        return frames, entry

# --------------------------
# Evaluation Function for LLaVA on HC-STVG
# --------------------------


def evaluate_entry_llava(frames, entry, runner, device, box_threshold=0.3, frame_step=15):
    """
    Run LLaVA on every `frame_step`-th frame, take only the first predicted bbox per frame,
    then compute mean vIoU and temporal IoU.

    LLaVA returns boxes as normalized [x1, y1, x2, y2].
    We convert them to absolute [x, y, w, h] using the current frame's size.
    """
    caption = entry.get("caption", "").strip()
    if not caption.endswith("(with grounding)"):
        caption += " (with grounding)"

    # Ground truth boxes from annotation in [x,y,w,h] format
    gt_boxes = entry.get("trajectory", [])
    # Sample ground truth boxes with the same step (if available)
    if len(gt_boxes) > 0:
        sampled_gt_boxes = gt_boxes[::frame_step]
    else:
        sampled_gt_boxes = []

    # Original dimensions from entry or first frame
    orig_W = entry.get("width")
    orig_H = entry.get("height")
    if orig_W is None or orig_H is None:
        if len(frames) > 0:
            orig_H, orig_W = frames[0].shape[:2]
        else:
            orig_W, orig_H = 800, 800

    predicted_boxes = []  # List of predicted boxes per processed frame

    # Process every frame_step-th frame
    for frame in frames[::frame_step]:
        frame_H, frame_W = frame.shape[:2]
        # Convert frame from BGR to RGB then save as temp image required by LLaVA
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.save(TEMP_IMAGE_PATH)

        # Run LLaVA inference; returns text and boxes (normalized [x1, y1, x2, y2])
        text, boxes = runner.run_inference(TEMP_IMAGE_PATH, caption)

        if boxes is not None and len(boxes) > 0:
            # Adjust boxes using unresize_boxes (using current frame dimensions)
            corrected_tensor = unresize_boxes(boxes, frame_W, frame_H)
            corrected_boxes = corrected_tensor.cpu().numpy()
            # Take the first box only.
            b = corrected_boxes[0]
            # Convert normalized [x1,y1,x2,y2] to absolute using current frame size
            x1 = round(b[0] * frame_W, 2)
            y1 = round(b[1] * frame_H, 2)
            x2 = round(b[2] * frame_W, 2)
            y2 = round(b[3] * frame_H, 2)
            # Convert to [x, y, w, h] format
            w_box = round(x2 - x1, 2)
            h_box = round(y2 - y1, 2)
            frame_box = [x1, y1, w_box, h_box]
        else:
            frame_box = [0, 0, 0, 0]
        predicted_boxes.append(frame_box)

    # Compute mean vIoU across the sampled frames
    frame_ious = []
    if len(sampled_gt_boxes) > 0:
        for pred_box, gt_box in zip(predicted_boxes, sampled_gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            frame_ious.append(iou)
        mv_iou = np.mean(frame_ious) if frame_ious else 0.0
    else:
        mv_iou = 0.0

    # Compute temporal IoU for the tube using tube start/end frames
    tube_start = entry.get('tube_start_frame', 0)
    tube_end = entry.get('tube_end_frame', 0)
    temporal_intersection = max(0, tube_end - tube_start)
    temporal_union = (tube_end - tube_start)
    mt_iou = temporal_intersection / temporal_union if temporal_union > 0 else 0.0

    return mv_iou, mt_iou, predicted_boxes

# --------------------------
# Main Evaluation Loop
# --------------------------


def main(args):
    runner = LlavaSingleSample(MODEL_PATH, VISION_CFG, INTER_CFG)
    dataset = HCSTVGDataloader(args.anno_path, args.video_dir)
    predictions_list = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        mv_iou, mt_iou, pred_boxes = evaluate_entry_llava(
            frames, entry, runner, device=args.device, frame_step=15)
        result = {
            "video_path": entry.get("video_path", ""),
            "caption": entry.get("caption", ""),
            "width": entry.get("width", None),
            "height": entry.get("height", None),
            "ground_truth_boxes": entry.get("trajectory", []),
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou,
            "mtIoU": mt_iou
        }
        predictions_list.append(result)
        print("Single Entry Evaluation:")
        print(f"Entry index: {args.entry_index}")
        print(f"mvIoU: {mv_iou:.4f}")
        print(f"mtIoU: {mt_iou:.4f}")
    else:
        iter_count = 0
        mvious, mtiou = [], []
        for frames, entry in tqdm(dataset, desc="Evaluating HC-STVG with LLaVA"):
            mv_iou, mt_iou, pred_boxes = evaluate_entry_llava(
                frames, entry, runner, device=args.device, frame_step=15)
            mvious.append(mv_iou)
            mtiou.append(mt_iou)
            result = {
                "video_path": entry.get("video_path", ""),
                "caption": entry.get("caption", ""),
                "width": entry.get("width", None),
                "height": entry.get("height", None),
                "ground_truth_boxes": entry.get("trajectory", []),
                "predicted_boxes": pred_boxes,
                "mvIoU": mv_iou,
                "mtIoU": mt_iou
            }
            predictions_list.append(result)
            iter_count += 1
            if args.max_iters > 0 and iter_count >= args.max_iters:
                break
        avg_mviou = np.mean(mvious) if mvious else 0.0
        avg_mtiou = np.mean(mtiou) if mtiou else 0.0
        print("Evaluation Results:")
        print(f"Mean mvIoU: {avg_mviou:.4f}")
        print(f"Mean mtIoU: {avg_mtiou:.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"Done! Predictions saved to {args.output_path}")


# --------------------------
# Argument Parsing and Execution
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLaVA Evaluation on HC-STVG Dataset")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to HC-STVG annotation JSON")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing HC-STVG videos")
    parser.add_argument("--output_path", type=str, default="./results/referral_hcstvg1_llava_predictions.json",
                        help="Path to save the predictions JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference (default: cuda)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (>= 0) or -1 for full dataset evaluation")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Maximum number of iterations (batches) to process (use a positive value for testing, -1 for full dataset)")

    args = parser.parse_args()
    main(args)
