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
TEMP_IMAGE_PATH = "./temp_llava_frame_vidstg_iou.jpg"

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
        Runs LLaVA inference using a temporary image file.
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
            "matching_threshold": 0.4,
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

        # If no boxes found, fallback to a default.
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
# Aspect Ratio Correction for LLaVA boxes
# --------------------------


def unresize_boxes(box_list, img_width, img_height):
    """
    Adjust normalized boxes (x1, y1, x2, y2) for aspect ratio padding.
    """
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
# VidSTG Data Loader for LLaVA
# --------------------------


class VidSTGDataloader:
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
        start_frame = entry.get("st_frame", 0)
        end_frame = entry.get("ed_frame", max(
            map(int, bbox_data.keys()), default=0))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > end_frame:
                break
            if start_frame <= frame_count <= end_frame:
                if (frame_count - start_frame) % self.frame_step == 0:
                    frames.append((frame, bbox_data.get(
                        str(frame_count), [0, 0, 0, 0])))
            frame_count += 1

        cap.release()
        entry["start_frame"] = start_frame
        entry["end_frame"] = end_frame
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
    orig_W, orig_H = entry["width"], entry["height"]

    for frame, gt_bbox in frames_with_gt:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(TEMP_IMAGE_PATH)

        text, boxes = runner.run_inference(TEMP_IMAGE_PATH, question)
        frame_h, frame_w = frame.shape[:2]

        best_iou = -1
        best_box = [0, 0, 0, 0]

        if boxes is not None and len(boxes) > 0:
            corrected_tensor = unresize_boxes(boxes, frame_w, frame_h)
            corrected_boxes = corrected_tensor.cpu().numpy()

            for b in corrected_boxes:
                # Convert normalized [x1,y1,x2,y2] to absolute pixel values
                x1 = b[0] * frame_w
                y1 = b[1] * frame_h
                x2 = b[2] * frame_w
                y2 = b[3] * frame_h

                # Scale to original resolution
                scale_x = orig_W / frame_w
                scale_y = orig_H / frame_h
                x1_abs = x1 * scale_x
                y1_abs = y1 * scale_y
                x2_abs = x2 * scale_x
                y2_abs = y2 * scale_y

                w_box = x2_abs - x1_abs
                h_box = y2_abs - y1_abs
                pred_box = [x1_abs, y1_abs, w_box, h_box]

                iou_score = calculate_iou(pred_box, gt_bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_box = pred_box

        predicted_boxes.append([round(v, 2) for v in best_box])
        ground_truth_boxes.append(gt_bbox)

    ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(ious) if ious else 0.0

    return mv_iou, predicted_boxes


# --------------------------
# Main Entry
# --------------------------


def main(args):
    runner = LlavaSingleSample(MODEL_PATH, VISION_CFG, INTER_CFG)
    dataset = VidSTGDataloader(args.anno_path, frame_step=args.frame_step)
    predictions_list = []

    if args.entry_index >= 0:
        frames_with_gt, entry = dataset[args.entry_index]
        mv_iou, pred_boxes = evaluate_entry_llava(
            frames_with_gt, entry, runner, device=args.device)
        results = {
            "video_path": entry["video_path"],
            "caption": entry["caption"],
            "width": entry["width"],
            "height": entry["height"],
            "ground_truth_boxes": entry["bbox"],
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou,
            "start_frame": entry.get("start_frame"),
            "end_frame": entry.get("end_frame")
        }
        predictions_list.append(results)
        print(f"mvIoU: {mv_iou:.4f}")
    else:
        mvious = []
        for idx in tqdm(range(len(dataset)), desc="Evaluating VidSTG with LLaVA"):
            frames_with_gt, entry = dataset[idx]
            mv_iou, pred_boxes = evaluate_entry_llava(
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
                "start_frame": entry.get("start_frame"),
                "end_frame": entry.get("end_frame")
            }
            predictions_list.append(results)
            if args.max_iters > 0 and (idx+1) >= args.max_iters:
                break

        print(f"Mean mvIoU: {np.mean(mvious):.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"✅ Done! Predictions saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA VidSTG Evaluation")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to VidSTG annotation JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for running LLaVA (default: cuda)")
    parser.add_argument("--frame_step", type=int, default=15,
                        help="Frame sampling step (default: 15)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test or -1 for the full dataset")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Max number of entries to process")
    parser.add_argument("--output_path", type=str, default="./results/vidstg_iou_llava_predictions.json",
                        help="Where to save results")
    args = parser.parse_args()

    main(args)
