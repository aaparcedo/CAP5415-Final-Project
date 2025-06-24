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
# Custom Dataset Loader for HC-STVG
# --------------------------


class HCSTVGDataloader:
    def __init__(self, annotation_path, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
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

        frames = []
        tube_start_frame = entry['tube_start_frame']
        tube_end_frame = entry['tube_end_frame']

        # Extract relevant frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > tube_end_frame:
                break
            if tube_start_frame <= frame_count <= tube_end_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame, _ = self.transform(frame, None)
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames, entry

# --------------------------
# IoU Calculation Functions
# --------------------------


def calculate_iou(box1, box2):
    """
    Computes IoU between two bounding boxes in [x, y, w, h] format.

    Args:
        box1 (list): First box in [x, y, w, h] format.
        box2 (list): Second box in [x, y, w, h] format.

    Returns:
        float: IoU score.
    """
    # Convert [x, y, w, h] → [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    # x2 = x1 + w, y2 = y1 + h
    box1_x2, box1_y2 = box1_x1 + box1[2], box1_y1 + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2_x1 + box2[2], box2_y1 + box2[3]

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate areas
    area1 = box1[2] * box1[3]  # width * height
    area2 = box2[2] * box2[3]

    # Compute IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --------------------------
# Evaluation Function
# --------------------------


def evaluate_entry(frames, entry, model, device, box_threshold=0.3, text_threshold=0.25):
    """
    Evaluates a single entry using the model and extracts the most confident bounding box for each frame.

    Args:
        frames (list): List of preprocessed frames (torch tensors).
        entry (dict): Video entry containing metadata and ground truth boxes.
        model (torch.nn.Module): The loaded model for evaluation.
        device (str): The device to run inference on ('cuda' or 'cpu').
        box_threshold (float): Confidence threshold for box selection.
        text_threshold (float): Text threshold for grounding.

    Returns:
        mv_iou (float): Mean vIoU across all frames.
        mt_iou (float): Mean tIoU for temporal alignment.
        predicted_boxes (list): List of predicted bounding boxes for each frame.
    """

    # Pre-process the caption
    caption = entry.get('caption', '')
    if isinstance(caption, list):
        caption = caption[0]
    caption = caption.strip().lower()
    if not caption:
        caption = "a person."
    if not caption.endswith('.'):
        caption += '.'

    # Ensure tube_start_frame and tube_end_frame are scalars
    tube_start = entry.get('tube_start_frame')
    if isinstance(tube_start, list):
        tube_start = tube_start[0]
    tube_end = entry.get('tube_end_frame')
    if isinstance(tube_end, list):
        tube_end = tube_end[0]

    ground_truth_boxes = entry['trajectory']
    predicted_boxes = []

    # Get the original image dimensions from the annotation
    orig_W = entry.get("width", 800)
    orig_H = entry.get("height", 800)

    for frame in frames:
        _, frame_H, frame_W = frame.shape

        # Run the model on the frame
        frame = frame.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(frame, captions=caption)

        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # Filter output using box confidence threshold
        # Get the highest confidence per box
        confidence_scores = logits.max(dim=1)[0]
        valid_mask = confidence_scores > box_threshold

        if valid_mask.sum() == 0:  # No valid boxes found
            predicted_boxes.append([0, 0, 0, 0])
            continue

        # Select the most confident box
        best_box_idx = confidence_scores.argmax()
        best_box = boxes[best_box_idx].detach().cpu().numpy()

        # Convert normalized (cx, cy, w, h) format to absolute (x1, y1, x2, y2)
        cx, cy, w, h = best_box
        x1 = (cx - w / 2) * frame_W
        y1 = (cy - h / 2) * frame_H
        x2 = (cx + w / 2) * frame_W
        y2 = (cy + h / 2) * frame_H

        # Scale coordinates to the original image dimensions
        scale_x = orig_W / frame_W
        scale_y = orig_H / frame_H
        x1_orig = x1 * scale_x
        y1_orig = y1 * scale_y
        x2_orig = x2 * scale_x
        y2_orig = y2 * scale_y
        predicted_box_abs = [x1_orig, y1_orig,
                             x2_orig-x1_orig, y2_orig-y1_orig]

        predicted_boxes.append(predicted_box_abs)

    # Compute IoU scores
    frame_ious = [calculate_iou(pred, gt) for pred, gt in zip(
        predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0

    temporal_intersection = max(0, (tube_end - tube_start))
    temporal_union = (tube_end - tube_start)
    mt_iou = temporal_intersection / temporal_union if temporal_union > 0 else 0

    return mv_iou, mt_iou, predicted_boxes


# --------------------------
# Custom Collate Function
# --------------------------


def custom_collate_fn(batch):
    return batch

# --------------------------
# Main Evaluation Loop
# --------------------------


def main(args):
    model = load_model(
        args.config_file, args.checkpoint_path, device=args.device)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = HCSTVGDataloader(
        args.anno_path, args.video_dir, transform=transform)

    predictions_list = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        mv_iou, mt_iou, pred_boxes = evaluate_entry(
            frames, entry, model, device=args.device)
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
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        mvious, mtiou, success_counts = [], [], {
            thresh: 0 for thresh in [0.1, 0.3, 0.5]}
        iter_count = 0
        for batch in tqdm(data_loader, desc="Evaluating HC-STVG-V2"):
            for frames, entry in batch:
                mv_iou, mt_iou, pred_boxes = evaluate_entry(
                    frames, entry, model, device=args.device)
                mvious.append(mv_iou)
                mtiou.append(mt_iou)
                for thresh in [0.1, 0.3, 0.5]:
                    if mv_iou >= thresh:
                        success_counts[thresh] += 1
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
        success_rates = {f"vIoU@{thresh}": (count / len(dataset))
                         for thresh, count in success_counts.items()}
        print("Evaluation Results:")
        print(f"Mean mvIoU: {avg_mviou:.4f}")
        print(f"Mean mtIoU: {avg_mtiou:.4f}")
        for thresh, rate in success_rates.items():
            print(f"{thresh}: {rate:.4%}")

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "referral_predictions_2.json")
    with open(output_file, "w") as f:
        json.dump(predictions_list, f, indent=4)
    print(f"Predictions saved to {output_file}")


# --------------------------
# Argument Parsing and Execution
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundingDINO Evaluation on HC-STVG")
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="Path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str,
                        required=True, help="Path to checkpoint file")
    parser.add_argument("--anno_path", type=str,
                        required=True, help="Path to annotation file")
    parser.add_argument("--video_dir", type=str,
                        required=True, help="Path to video directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation (default: cuda)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (>= 0) or -1 for full dataset evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for full dataset evaluation")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Maximum number of iterations (batches) to process (use a positive value for testing, -1 for full dataset)")
    args = parser.parse_args()

    main(args)
