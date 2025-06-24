import argparse
import json
import os
import cv2
import numpy as np
import torch
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
TEMP_IMAGE_PATH = "./temp_llava_frame_se.jpg"

# --------------------------
# LLaVA Wrapper
# --------------------------

class LlavaSingleSample:
    def __init__(self):
        self.model_backend = Evaluator_MM_Inter(
            model_path=MODEL_PATH,
            path_vision_model_cfg=VISION_CFG,
            path_inter_model_cfg=INTER_CFG
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
# Data Loader
# --------------------------

class SomethingElseLoader:
    def __init__(self, annotation_path, frame_step=15):
        with open(annotation_path, "r") as f:
            self.data = json.load(f)
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
                    bbox = bbox_data[str(frame_count)]  # [x1, y1, x2, y2]
                    frames.append((frame, bbox))

            frame_count += 1

        cap.release()
        return frames, entry

# --------------------------
# IoU calculation (x1, y1, x2, y2)
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
# Unresize LLaVA boxes
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
# Evaluate Entry
# --------------------------

def evaluate_entry_llava(frames_with_gt, entry, runner):
    question = entry.get("caption", "a person").strip()
    if not question.endswith("(with grounding)"):
        question += " (with grounding)"

    predicted_boxes = []
    ground_truth_boxes = []
    orig_W = entry.get("width")
    orig_H = entry.get("height")

    for frame, gt_bbox in frames_with_gt:
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
            
            predicted_boxes.append([x1, y1, x2, y2])
        else:
            predicted_boxes.append([0, 0, 0, 0])

        ground_truth_boxes.append(gt_bbox)

    ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
    mv_iou = np.mean(ious) if ious else 0.0

    return mv_iou, predicted_boxes

# --------------------------
# Main
# --------------------------

def main(args):
    runner = LlavaSingleSample()
    dataset = SomethingElseLoader(args.anno_path, frame_step=args.frame_step)
    predictions = []

    if args.entry_index >= 0:
        frames_with_gt, entry = dataset[args.entry_index]
        mv_iou, pred_boxes = evaluate_entry_llava(
            frames_with_gt, entry, runner)

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
        for idx in tqdm(range(len(dataset)), desc="Evaluating Something-Else with LLaVA"):
            frames_with_gt, entry = dataset[idx]
            if len(frames_with_gt) == 0:
                print(f"[Skip] No valid frames for sample {idx}")
                continue

            mv_iou, pred_boxes = evaluate_entry_llava(
                frames_with_gt, entry, runner)
            mv_ious.append(mv_iou)

            result = {
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "ground_truth_boxes": entry["bbox"],
                "predicted_boxes": pred_boxes,
                "mvIoU": mv_iou
            }
            predictions.append(result)

            if args.max_iters > 0 and (idx + 1) >= args.max_iters:
                break

        print(f"Mean mvIoU over {len(mv_ious)} samples: {np.mean(mv_ious):.4f}")

    os.makedirs("output_results", exist_ok=True)
    output_file = os.path.join(
        "output_results", "somethingelse_llava_predictions.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predictions saved to {output_file}")

# --------------------------
# Arguments
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLaVA Evaluation on Something-Else Dataset")
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--frame_step", type=int, default=15)
    parser.add_argument("--entry_index", type=int, default=-1)
    parser.add_argument("--max_iters", type=int, default=-1)
    args = parser.parse_args()

    main(args)
