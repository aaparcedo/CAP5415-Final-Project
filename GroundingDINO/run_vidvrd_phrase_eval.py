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
from groundingdino.util.utils import get_phrases_from_posmap

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)

class VidVRDDataloader:
    def __init__(self, annotation_path, transform=None, frame_stride=15):
        self.transform = transform
        self.frame_stride = frame_stride
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []
        frame_indices = []
        st_frame = entry['st_frame']
        ed_frame = entry['ed_frame']

        sampled_ids = set(range(st_frame, ed_frame + 1, self.frame_stride))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_id > ed_frame:
                break
            if frame_id in sampled_ids:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                if self.transform:
                    frame_tensor, _ = self.transform(frame_pil, None)
                    frames.append(frame_tensor)
                    frame_indices.append(frame_id)
            frame_id += 1

        cap.release()
        return frames, frame_indices, entry


def get_grounding_output(model, frame, caption, box_threshold=0.001, text_threshold=0.001, device="cuda"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    frame = frame.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(frame, captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    confidence_scores = logits.max(dim=1)[0]
    valid_mask = confidence_scores > box_threshold

    boxes = boxes[valid_mask]
    scores = confidence_scores[valid_mask]
    logits = logits[valid_mask]

    tokenized = model.tokenizer(caption)
    pred_phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer)
        for logit in logits
    ]

    return boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), pred_phrases


def calculate_iou(box1, box2):
    """
    Computes IoU between two bounding boxes.
    box1: [x, y, w, h]
    box2: [x1, y1, x2, y2]

    Returns:
        float: IoU score.
    """
    # Convert box1 to [x1, y1, x2, y2]
    box1_x1 = box1[0]
    box1_y1 = box1[1]
    box1_x2 = box1_x1 + box1[2]
    box1_y2 = box1_y1 + box1[3]

    # Use box2 as is
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Areas of both boxes
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Compute IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def evaluate_entry(frames, frame_indices, entry, model, device, box_threshold, text_threshold):
    orig_W = entry.get("width", 800)
    orig_H = entry.get("height", 800)
    video_id = entry["video_path"].split("/")[-1].replace(".mp4", "")

    caption = entry.get('caption', '').strip().lower()
    if not caption.endswith('.'):
        caption += '.'

    all_results = []

    for i, obj_name in enumerate(entry["placeholders"]):
        object_name = obj_name.strip().lower()
        category_id = entry["category_id_map"][object_name]

        # Determine GT boxes: assume first placeholder is "subject", second is "object"
        gt_boxes_raw = entry["bboxes"]["subject" if i == 0 else "object"]
        gt_box_map = {int(k): v for k, v in gt_boxes_raw.items()}

        predicted_boxes_per_frame = []

        for frame_tensor, frame_number in zip(frames, frame_indices):
            _, frame_H, frame_W = frame_tensor.shape

            top_boxes, top_scores, pred_phrases = get_grounding_output(
                model, frame_tensor, caption, box_threshold, text_threshold, device=device)

            best_match = None
            best_score = -1
            predicted_phrase = None
            for box, score, phrase in zip(top_boxes, top_scores, pred_phrases):
                print(f"Current phrase:{phrase.lower()}")
                print(f"Current object name:{object_name}")
                if object_name in phrase.lower() and score > best_score:
                    best_match = box
                    best_score = score
                    predicted_phrase = phrase.lower()

            if best_match is not None:
                cx, cy, w, h = best_match
                x1 = (cx - w / 2) * frame_W
                y1 = (cy - h / 2) * frame_H
                bw = w * frame_W
                bh = h * frame_H
                x1 *= orig_W / frame_W
                y1 *= orig_H / frame_H
                bw *= orig_W / frame_W
                bh *= orig_H / frame_H

                predicted_boxes_per_frame.append({
                    "frame_idx": frame_number,
                    "bbox": [x1, y1, bw, bh],
                    "score": float(best_score),
                    "predicted_phrase": predicted_phrase
                })

        # Compute mvIoU
        pred_box_map = {d['frame_idx']: d['bbox'] for d in predicted_boxes_per_frame}
        common_frames = sorted(set(gt_box_map.keys()) & set(pred_box_map.keys()))
        frame_ious = [
            calculate_iou(pred_box_map[f], gt_box_map[f]) for f in common_frames
        ]
        mv_iou = np.mean(frame_ious) if frame_ious else 0.0

        # Save result for this object
        all_results.append({
            "object_name": object_name,
            "category_id": category_id,
            "predicted_boxes_per_frame": predicted_boxes_per_frame,
            "ground_truth_boxes": gt_box_map,
            "mv_iou": mv_iou
        })

    # Bundle into one raw_result
    raw_result = {
        "video_path": entry.get("video_path", ""),
        "caption": caption,
        "width": orig_W,
        "height": orig_H,
        "tube_start_frame": entry.get("st_frame"),
        "tube_end_frame": entry.get("ed_frame"),
        "results_per_object": all_results
    }

    return raw_result


def main(args):
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = VidVRDDataloader(args.anno_path, transform=transform, frame_stride=15)
    raw_results = []
    all_mv_ious = []

    if args.entry_index >= 0:
        idx = args.entry_index
        frames, frame_indices, entry = dataset[idx]
        entry["video_id"] = idx
        raw_result = evaluate_entry(
            frames, frame_indices, entry, model, args.device,
            args.box_threshold, args.text_threshold)
        raw_results.append(raw_result)

        for obj_result in raw_result["results_per_object"]:
            all_mv_ious.append(obj_result["mv_iou"])
    else:
        for idx in tqdm(range(len(dataset)), desc="Evaluating VidVRD dataset"):
            frames, frame_indices, entry = dataset[idx]
            entry["video_id"] = idx
            raw_result = evaluate_entry(
                frames, frame_indices, entry, model, args.device,
                args.box_threshold, args.text_threshold)
            raw_results.append(raw_result)

            for obj_result in raw_result["results_per_object"]:
                all_mv_ious.append(obj_result["mv_iou"])

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    raw_output_file = os.path.join(output_dir, "vidvrd_phrase_predictions_iou.json")
    with open(raw_output_file, "w") as f:
        json.dump(raw_results, f, indent=4)

    print(f"Saved raw prediction results to {raw_output_file}")

    # vIoU Evaluation
    mean_mv_iou = np.mean(all_mv_ious) if all_mv_ious else 0.0
    viou_at_03 = sum(iou >= 0.3 for iou in all_mv_ious) / len(all_mv_ious) if all_mv_ious else 0.0
    viou_at_05 = sum(iou >= 0.5 for iou in all_mv_ious) / len(all_mv_ious) if all_mv_ious else 0.0

    print("\n--- vIoU Evaluation ---")
    print(f"Mean mvIoU: {mean_mv_iou:.4f}")
    print(f"vIoU@0.3: {viou_at_03:.4f}")
    print(f"vIoU@0.5: {viou_at_05:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundingDINO phrase grounding eval (VidVRD format)")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--entry_index", type=int, default=-1, help="Evaluate a specific entry index only")
    args = parser.parse_args()

    main(args)
