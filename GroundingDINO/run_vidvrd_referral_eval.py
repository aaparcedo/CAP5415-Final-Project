import argparse
import json
import os
from PIL import Image
import torch
import groundingdino.datasets.transforms as T
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
# Dataset Loader for VidVRD (with frame_step)
# --------------------------

class VidVRDDataloader:
    def __init__(self, annotation_path, transform=None, frame_step=15):
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
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
        bbox_data = entry.get("bbox", {})  # frame number -> bbox
        valid_frame_ids = sorted([int(k) for k in bbox_data.keys()])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if str(frame_count) in bbox_data and frame_count % self.frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                if self.transform:
                    frame_tensor, _ = self.transform(frame_pil, None)
                else:
                    frame_tensor = frame_pil  # unlikely

                bbox = bbox_data[str(frame_count)]  # [xmin, ymin, xmax, ymax]
                frames.append((frame_tensor, bbox, frame_count))

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            print(f"[Warning] No valid frames found for video {video_path}")

        return frames, entry

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
# Evaluation per Entry
# --------------------------

def evaluate_entry(frames, entry, model, device, box_threshold=0.3):
    caption = entry.get("caption", "a person.")
    if not caption.endswith("."):
        caption += "."

    ground_truth_boxes = [bbox for _, bbox, _ in frames]
    predicted_boxes = []
    frame_ids = [fid for _, _, fid in frames]

    orig_W, orig_H = entry["width"], entry["height"]

    for frame_tensor, gt_box, frame_id in frames:
        _, H, W = frame_tensor.shape  # C, H, W
        frame_input = frame_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame_input, captions=caption)

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        confidence_scores = logits.max(dim=1)[0]
        valid_mask = confidence_scores > box_threshold

        if valid_mask.sum() == 0:
            predicted_boxes.append([0, 0, 0, 0])
            continue

        best_box_idx = confidence_scores.argmax()
        best_box = boxes[best_box_idx].detach().cpu().numpy()

        # Normalized cx, cy, w, h --> pixel xmin, ymin, xmax, ymax
        cx, cy, w, h = best_box
        xmin = (cx - w / 2) * W
        ymin = (cy - h / 2) * H
        xmax = (cx + w / 2) * W
        ymax = (cy + h / 2) * H

        # Scale to original video resolution
        scale_x = orig_W / W
        scale_y = orig_H / H

        pred_abs = [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]
        predicted_boxes.append(pred_abs)

    frame_ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0

    return mv_iou, predicted_boxes, frame_ids

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

    dataset = VidVRDDataloader(args.anno_path, transform=transform, frame_step=args.frame_step)
    predictions_list = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        mv_iou, pred_boxes, frame_ids = evaluate_entry(
            frames, entry, model, device=args.device)

        result = {
            "video_path": entry["video_path"],
            "caption": entry["caption"],
            "width": entry["width"],
            "height": entry["height"],
            "ground_truth_boxes": entry["bbox"],
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou,
            "frame_ids": frame_ids
        }
        predictions_list.append(result)
        print(f"[Single Entry] mvIoU: {mv_iou:.4f}")

    else:
        mv_ious = []

        for idx in tqdm(range(len(dataset)), desc="Evaluating VidVRD"):
            frames, entry = dataset[idx]

            if len(frames) == 0:
                print(f"[Skip] No valid frames for sample {idx}")
                continue

            mv_iou, pred_boxes, frame_ids = evaluate_entry(
                frames, entry, model, device=args.device)
            mv_ious.append(mv_iou)

            result = {
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "width": entry["width"],
                "height": entry["height"],
                "ground_truth_boxes": entry["bbox"],
                "predicted_boxes": pred_boxes,
                "mvIoU": mv_iou,
                "frame_ids": frame_ids
            }
            predictions_list.append(result)

            if args.max_iters > 0 and idx + 1 >= args.max_iters:
                break

        print(f"Mean mvIoU over {len(mv_ious)} samples: {np.mean(mv_ious):.4f}")

    os.makedirs("output_results", exist_ok=True)
    output_file = os.path.join(
        "output_results", "vidvrd_predictions_referral_full.json")
    with open(output_file, "w") as f:
        json.dump(predictions_list, f, indent=4)

    print(f"Predictions saved to {output_file}")

# --------------------------
# Argument Parsing
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundingDINO Evaluation on VidVRD (subject bbox only, with frame jump)")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entry_index", type=int, default=-1)
    parser.add_argument("--frame_step", type=int, default=15,
                        help="Frame sampling step (default: 15)")
    parser.add_argument("--max_iters", type=int, default=-1)
    args = parser.parse_args()

    main(args)
