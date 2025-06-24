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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

############################################
# Load Model
############################################

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)

############################################
# HC-STVG Dataloader — 15 frame jump
############################################

class HCSTVGDataloader:
    def __init__(self, annotation_path, video_dir, transform=None, frame_stride=15):
        self.video_dir = video_dir
        self.transform = transform
        self.frame_stride = frame_stride
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
        frame_indices = []
        tube_start = entry['tube_start_frame']
        tube_end = entry['tube_end_frame']

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count > tube_end:
                break

            # Sample every N frames in the tube
            if frame_count >= tube_start and ((frame_count - tube_start) % self.frame_stride == 0):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                if self.transform:
                    frame_tensor, _ = self.transform(frame_pil, None)
                    frames.append(frame_tensor)
                    frame_indices.append(frame_count)

            frame_count += 1

        cap.release()

        return frames, frame_indices, entry

############################################
# Inference per Frame
############################################

def get_grounding_output(model, frame, caption, box_threshold=0.3, text_threshold=0.25, topk=5, device="cuda"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    frame = frame.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(frame, captions=caption)

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    confidence_scores = logits.max(dim=1)[0]
    valid_mask = confidence_scores > box_threshold

    boxes = boxes[valid_mask]
    scores = confidence_scores[valid_mask]

    if len(scores) == 0:
        return [], []

    sorted_idx = scores.argsort(descending=True)[:topk]
    top_boxes = boxes[sorted_idx].detach().cpu().numpy()
    top_scores = scores[sorted_idx].detach().cpu().numpy()

    return top_boxes, top_scores

############################################
# Evaluate a single video
############################################

def evaluate_entry(frames, frame_indices, entry, model, device, box_threshold, text_threshold, topk=5):
    caption = entry.get('caption', '')
    if isinstance(caption, list):
        caption = caption[0]
    caption = caption.strip().lower()
    if not caption:
        caption = "a person."
    if not caption.endswith('.'):
        caption += '.'

    ground_truth_boxes = entry['trajectory']

    orig_W = entry.get("width", 800)
    orig_H = entry.get("height", 800)

    predicted_boxes_per_frame = []
    coco_predictions = []

    for frame_tensor, frame_number in zip(frames, frame_indices):
        _, frame_H, frame_W = frame_tensor.shape

        top_boxes, top_scores = get_grounding_output(
            model, frame_tensor, caption, box_threshold, text_threshold, topk=topk, device=device)

        box_list = []
        score_list = []

        for box, score in zip(top_boxes, top_scores):
            cx, cy, w, h = box
            x1 = (cx - w / 2) * frame_W
            y1 = (cy - h / 2) * frame_H
            bw = w * frame_W
            bh = h * frame_H

            # Scale to original video size
            scale_x = orig_W / frame_W
            scale_y = orig_H / frame_H

            x1 *= scale_x
            y1 *= scale_y
            bw *= scale_x
            bh *= scale_y

            box_list.append([x1, y1, bw, bh])
            score_list.append(float(score))

            coco_predictions.append({
                "image_id": entry["video_id"] * 10000 + frame_number,  # Unique ID per video/frame
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "score": float(score)
            })

        predicted_boxes_per_frame.append({
            "frame_idx": frame_number,
            "boxes": box_list,
            "scores": score_list
        })

    result = {
        "video_path": entry.get("video_path", ""),
        "caption": entry.get("caption", ""),
        "width": orig_W,
        "height": orig_H,
        "tube_start_frame": entry.get("tube_start_frame"),
        "tube_end_frame": entry.get("tube_end_frame"),
        "ground_truth_boxes": ground_truth_boxes,
        "predicted_boxes_per_frame": predicted_boxes_per_frame
    }

    return result, coco_predictions

############################################
# Main
############################################

def main(args):
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = HCSTVGDataloader(
        args.anno_path, args.video_dir, transform=transform, frame_stride=15)

    results_list = []
    coco_preds = []

    if args.entry_index >= 0:
        frames, frame_indices, entry = dataset[args.entry_index]
        entry["video_id"] = args.entry_index  # assign video ID
        result, coco_predictions = evaluate_entry(
            frames, frame_indices, entry, model, args.device,
            args.box_threshold, args.text_threshold, topk=5)
        results_list.append(result)
        coco_preds.extend(coco_predictions)
    else:
        for idx in tqdm(range(len(dataset)), desc="Evaluating dataset"):
            frames, frame_indices, entry = dataset[idx]
            entry["video_id"] = idx  # assign video ID
            result, coco_predictions = evaluate_entry(
                frames, frame_indices, entry, model, args.device,
                args.box_threshold, args.text_threshold, topk=5)
            results_list.append(result)
            coco_preds.extend(coco_predictions)

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    raw_output_file = os.path.join(output_dir, "hcstvg2_phrase_predictions_raw.json")
    with open(raw_output_file, "w") as f:
        json.dump(results_list, f, indent=4)

    coco_output_file = os.path.join(output_dir, "hcstvg2_phrase_predictions_coco.json")
    with open(coco_output_file, "w") as f:
        json.dump(coco_preds, f)

    print(f"Saved detailed predictions to {raw_output_file}")
    print(f"Saved COCO predictions to {coco_output_file}")

    ############################################
    # COCO Evaluation
    ############################################

    print("\nRunning COCO Evaluation:")
    coco_gt = COCO(args.coco_gt)
    coco_dt = coco_gt.loadRes(coco_output_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

############################################
# Argument Parsing
############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundingDINO phrase grounding evaluation on HC-STVG with COCO metrics (sampling every 15 frames)")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--coco_gt", type=str, required=True, help="Path to COCO-format ground truth JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--entry_index", type=int, default=-1)
    args = parser.parse_args()

    main(args)
    
