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
# CONFIG
# --------------------------

MODEL_PATH = "/home/da530038/llava-grouding/LLaVA-Grounding/checkpoints/llava_grounding"
VISION_CFG = "configs/openseed/openseed_swint_lang_joint_2st_visual_prompt.yaml"
INTER_CFG = "configs/semsam/visual_prompt_encoder.yaml"
TEMP_IMAGE_PATH = "./temp_llava_frame_vidvrd.jpg"

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
        input_data = {
            "file_name": image_path,
            "image_id": 0,
            "question_id": 0,
            "conversations": [[[{"from": "human", "value": DEFAULT_IMAGE_TOKEN + " " + question},
                               {"from": "gpt", "value": "Placeholder."}], None]],
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
# IoU Calculation
# --------------------------

def calculate_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# --------------------------
# Aspect Ratio Correction for LLaVA boxes
# --------------------------

def unresize_boxes(box_list, img_width, img_height):
    if not box_list:
        return []
    box_tensor = torch.stack(box_list).squeeze(1).clone()
    ratio = min(img_width, img_height) / max(img_width, img_height)
    if img_width > img_height:
        box_tensor[:, 1] /= ratio
        box_tensor[:, 3] /= ratio
    elif img_width < img_height:
        box_tensor[:, 0] /= ratio
        box_tensor[:, 2] /= ratio
    return box_tensor

# --------------------------
# VidVRD Data Loader (with frame_step)
# --------------------------

class VidVRDDataloader:
    def __init__(self, annotation_path, frame_step=15):
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)
        self.frame_step = frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []
        bbox_data = entry.get("bbox", {})
        frame_ids = sorted([int(k) for k in bbox_data.keys()])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample only frames with GT and at frame_step interval
            if str(frame_count) in bbox_data and frame_count % self.frame_step == 0:
                bbox = bbox_data[str(frame_count)]
                frames.append((frame, bbox, frame_count))

            frame_count += 1

        cap.release()

        return frames, entry

# --------------------------
# Evaluate Single Entry with LLaVA
# --------------------------

def evaluate_entry_llava(frames_with_gt, entry, runner, device="cuda"):
    question = entry.get("caption", "a person").strip()
    if not question.endswith("(with grounding)"):
        question += " (with grounding)"

    predicted_boxes = []
    ground_truth_boxes = []
    frame_indices = []
    orig_W, orig_H = entry["width"], entry["height"]

    for frame, gt_bbox, frame_id in frames_with_gt:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(TEMP_IMAGE_PATH)

        text, boxes = runner.run_inference(TEMP_IMAGE_PATH, question)

        frame_h, frame_w = frame.shape[:2]
        if boxes is not None and len(boxes) > 0:
            corrected_tensor = unresize_boxes(boxes, frame_w, frame_h)
            corrected_boxes = corrected_tensor.cpu().numpy()
            b = corrected_boxes[0]
            x1 = b[0] * frame_w
            y1 = b[1] * frame_h
            x2 = b[2] * frame_w
            y2 = b[3] * frame_h

            scale_x = orig_W / frame_w
            scale_y = orig_H / frame_h

            xmin = x1 * scale_x
            ymin = y1 * scale_y
            xmax = x2 * scale_x
            ymax = y2 * scale_y

            predicted_boxes.append([xmin, ymin, xmax, ymax])
        else:
            predicted_boxes.append([0, 0, 0, 0])

        ground_truth_boxes.append(gt_bbox)
        frame_indices.append(frame_id)

    ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(ious) if ious else 0

    return mv_iou, predicted_boxes, frame_indices

# --------------------------
# Main Evaluation Loop
# --------------------------

def main(args):
    runner = LlavaSingleSample(MODEL_PATH, VISION_CFG, INTER_CFG)
    dataset = VidVRDDataloader(args.anno_path, frame_step=args.frame_step)
    predictions_list = []

    if args.entry_index >= 0:
        frames_with_gt, entry = dataset[args.entry_index]
        mv_iou, pred_boxes, frame_ids = evaluate_entry_llava(
            frames_with_gt, entry, runner, device=args.device)
        results = {
            "video_path": entry["video_path"],
            "caption": entry["caption"],
            "width": entry["width"],
            "height": entry["height"],
            "ground_truth_boxes": entry["bbox"],
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou,
            "frame_ids": frame_ids
        }
        predictions_list.append(results)
        print(f"[Single Entry] mvIoU: {mv_iou:.4f}")

    else:
        mvious = []
        for idx in tqdm(range(len(dataset)), desc="Evaluating VidVRD with LLaVA"):
            frames_with_gt, entry = dataset[idx]
            if len(frames_with_gt) == 0:
                print(f"[Skip] No valid frames for sample {idx}")
                continue

            mv_iou, pred_boxes, frame_ids = evaluate_entry_llava(
                frames_with_gt, entry, runner, device=args.device)
            mvious.append(mv_iou)

            results = {
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "width": entry["width"],
                "height": entry["height"],
                "ground_truth_boxes": entry["bbox"],
                "predicted_boxes": pred_boxes,
                "mvIoU": mv_iou,
                "frame_ids": frame_ids
            }
            predictions_list.append(results)

            if args.max_iters > 0 and (idx+1) >= args.max_iters:
                break

        print(f"Mean mvIoU over {len(mvious)} samples: {np.mean(mvious):.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"✅ Done! Predictions saved to {args.output_path}")

# --------------------------
# Argument Parsing
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA VidVRD Evaluation (subject only, frame sampling supported)")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to VidVRD annotation JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for running LLaVA")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (-1 for all)")
    parser.add_argument("--frame_step", type=int, default=15,
                        help="Frame sampling step (default: 15)")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Max number of entries to process")
    parser.add_argument("--output_path", type=str, default="./results/vidvrd_llava_predictions.json",
                        help="Where to save predictions")
    args = parser.parse_args()

    main(args)
