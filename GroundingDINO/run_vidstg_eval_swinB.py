import argparse
import json
import os
from PIL import Image
import torch
import groundingdino.datasets.transforms as T
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import clean_state_dict

# --------------------------
# Model Loading Function
# --------------------------


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)

# --------------------------
# Custom Dataset Loader for VidSTG
# --------------------------


class VidSTGDataloader:
    def __init__(self, annotation_path, transform=None):
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)  # Ensure it's a list
        self.transform = transform

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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame, _ = self.transform(frame, None)

                bbox = bbox_data.get(str(frame_count), [0, 0, 0, 0])
                frames.append((frame, bbox))
            frame_count += 1

        cap.release()
        return frames, entry

# --------------------------
# IoU Calculation Functions
# --------------------------


def calculate_iou(box1, box2):
    """
    Compute IoU between two bounding boxes in (x, y, w, h) format.
    """
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# --------------------------
# Evaluation Function
# --------------------------


def evaluate_entry(frames, entry, model, device, box_threshold=0.3):
    caption = entry.get("caption", "a person.")
    caption = caption.strip().lower()
    if not caption.endswith("."):
        caption += "."

    ground_truth_boxes = [box for _, box in frames]
    predicted_boxes = []

    orig_W, orig_H = entry["width"], entry["height"]

    for frame, _ in frames:
        _, frame_H, frame_W = frame.shape
        frame = frame.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame, captions=caption)

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        confidence_scores = logits.max(dim=1)[0]
        valid_mask = confidence_scores > box_threshold

        if valid_mask.sum() == 0:
            predicted_boxes.append([0, 0, 0, 0])
            continue

        best_box_idx = confidence_scores.argmax()
        best_box = boxes[best_box_idx].detach().cpu().numpy()

        # Convert normalized box to absolute coordinates
        cx, cy, w, h = best_box
        x1 = (cx - w / 2) * frame_W
        y1 = (cy - h / 2) * frame_H
        x2 = (cx + w / 2) * frame_W
        y2 = (cy + h / 2) * frame_H

        scale_x = orig_W / frame_W
        scale_y = orig_H / frame_H

        predicted_box_abs = [x1 * scale_x, y1 * scale_y,
                             (x2-x1) * scale_x, (y2-y1) * scale_y]
        predicted_boxes.append(predicted_box_abs)

    frame_ious = [calculate_iou(pred, gt) for pred, gt in zip(
        predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0

    return mv_iou, predicted_boxes

# --------------------------
# Main Evaluation Loop
# --------------------------


def main(args):
    model = load_model(
        args.config_file, args.checkpoint_path, device=args.device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = VidSTGDataloader(args.anno_path, transform=transform)
    predictions_list = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        mv_iou, pred_boxes = evaluate_entry(
            frames, entry, model, device=args.device)

        result = {
            "video_path": entry["video_path"],
            "caption": entry["caption"],
            "width": entry["width"],
            "height": entry["height"],
            "ground_truth_boxes": entry["bbox"],
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou,
        }
        predictions_list.append(result)
        print(f"Single Entry Evaluation:\nmvIoU: {mv_iou:.4f}")

    else:
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x))
        )
        mvious = []
        iter_count = 0

        for frames_list, entries_list in tqdm(data_loader, desc="Evaluating VidSTG"):
            for frames, entry in zip(frames_list, entries_list):
                mv_iou, pred_boxes = evaluate_entry(
                    frames, entry, model, device=args.device)
                mvious.append(mv_iou)

                result = {
                    "video_path": entry["video_path"],
                    "caption": entry["caption"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "ground_truth_boxes": entry["bbox"],
                    "predicted_boxes": pred_boxes,
                    "mvIoU": mv_iou,
                }
                predictions_list.append(result)

            iter_count += 1
            if args.max_iters > 0 and iter_count >= args.max_iters:
                break

        print(f"Mean mvIoU: {np.mean(mvious):.4f}")

    os.makedirs("output_results", exist_ok=True)
    output_file = os.path.join(
        "output_results", "vidstg_predictions_full_swinB.json")
    with open(output_file, "w") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"Predictions saved to {output_file}")

# --------------------------
# Argument Parsing and Execution
# --------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundingDINO Evaluation on VidSTG")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entry_index", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=-1)
    args = parser.parse_args()

    main(args)
