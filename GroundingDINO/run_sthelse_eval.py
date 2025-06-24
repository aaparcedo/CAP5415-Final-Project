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
# Model Loading
# --------------------------

def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)

# --------------------------
# Data Loader
# --------------------------

class SomethingElseLoader:
    def __init__(self, annotation_path, transform=None, frame_step=15):
        with open(annotation_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform
        self.frame_step = frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        bbox_data = entry["bbox"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        frame_ids = sorted([int(k) for k in bbox_data.keys()])
        st_frame = entry.get("st_frame", min(frame_ids))
        ed_frame = entry.get("ed_frame", max(frame_ids))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > ed_frame:
                break

            if frame_count in frame_ids:
                if (frame_count - st_frame) % self.frame_step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    if self.transform:
                        pil_frame, _ = self.transform(pil_frame, None)
                    bbox = bbox_data[str(frame_count)]  # [x1, y1, x2, y2]
                    frames.append((pil_frame, bbox))

            frame_count += 1

        cap.release()
        return frames, entry

# --------------------------
# IoU calculation (x1, y1, x2, y2 format)
# --------------------------

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# --------------------------
# Evaluate a sample
# --------------------------

def evaluate_entry(frames, entry, model, device, box_threshold=0.3):
    caption = entry.get('caption', '')
    if isinstance(caption, list):
        caption = caption[0]
    caption = caption.strip().lower()
    if not caption:
        caption = "a person."
    if not caption.endswith('.'):
        caption += '.'

    gt_boxes = [bbox for _, bbox in frames]  # already [x1, y1, x2, y2]
    pred_boxes = []

    orig_W = entry.get("width", 456)
    orig_H = entry.get("height", 256)  # Something-Else height

    for frame, gt in frames:
        _, H, W = frame.shape  # C, H, W
        frame_input = frame.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame_input, captions=caption)

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        scores = logits.max(dim=1)[0]
        valid_mask = scores > box_threshold

        if valid_mask.sum() == 0:
            pred_boxes.append([0, 0, 0, 0])
            continue

        best_idx = scores.argmax()
        best_box = boxes[best_idx].detach().cpu().numpy()

        cx, cy, w, h = best_box
        xmin = (cx - w / 2) * W
        ymin = (cy - h / 2) * H
        xmax = (cx + w / 2) * W
        ymax = (cy + h / 2) * H

        # Scale to original video size (if resized)
        scale_x = orig_W / W
        scale_y = orig_H / H

        xmin *= scale_x
        ymin *= scale_y
        xmax *= scale_x
        ymax *= scale_y

        pred_abs = [xmin, ymin, xmax, ymax]  # match GT format
        pred_boxes.append(pred_abs)

    ious = [calculate_iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]
    mv_iou = np.mean(ious) if ious else 0.0

    return mv_iou, pred_boxes

# --------------------------
# Main
# --------------------------

def main(args):
    model = load_model(
        args.config_file, args.checkpoint_path, device=args.device)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = SomethingElseLoader(
        args.anno_path, transform=transform, frame_step=args.frame_step)
    predictions = []

    if args.entry_index >= 0:
        frames, entry = dataset[args.entry_index]
        mv_iou, pred_boxes = evaluate_entry(
            frames, entry, model, device=args.device)

        result = {
            "video_path": entry["video_path"],
            "caption": entry["caption"],
            "ground_truth_boxes": entry["bbox"],
            "predicted_boxes": pred_boxes,
            "mvIoU": mv_iou
        }
        predictions.append(result)
        print(f"[Single Entry] mvIoU: {mv_iou:.4f}")

    else:
        mv_ious = []
        for idx in tqdm(range(len(dataset)), desc="Evaluating Something-Else"):
            frames, entry = dataset[idx]
            if len(frames) == 0:
                print(f"[Skip] No valid frames for sample {idx}")
                continue

            mv_iou, pred_boxes = evaluate_entry(
                frames, entry, model, device=args.device)
            mv_ious.append(mv_iou)

            result = {
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "ground_truth_boxes": entry["bbox"],
                "predicted_boxes": pred_boxes,
                "mvIoU": mv_iou
            }
            predictions.append(result)

            if args.max_iters > 0 and idx + 1 >= args.max_iters:
                break

        print(f"Mean mvIoU over {len(mv_ious)} samples: {np.mean(mv_ious):.4f}")

    os.makedirs("output_results", exist_ok=True)
    output_file = os.path.join(
        "output_results", "somethingelse_predictions.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predictions saved to {output_file}")

# --------------------------
# Arguments
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundingDINO Evaluation on Something-Else Dataset")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--entry_index", type=int, default=-1)
    parser.add_argument("--max_iters", type=int, default=-1)
    parser.add_argument("--frame_step", type=int, default=15)
    args = parser.parse_args()

    main(args)
