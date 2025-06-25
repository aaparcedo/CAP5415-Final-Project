import json
import numpy as np
import pandas as pd


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])  # Convert width to x2
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])  # Convert height to y2

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def evaluate_llava_predictions(json_path, frame_step=15):
    """Evaluate LLaVA predictions using frame-stepped sampling and ground truth dict."""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_video_ious = []
    vIoU_03 = 0
    vIoU_05 = 0
    total_videos = 0

    for sample in data:
        gt_dict = sample["ground_truth_boxes"]
        pred_boxes = sample["predicted_boxes"]

        if not gt_dict or not pred_boxes:
            continue

        gt_keys = sorted(gt_dict.keys(), key=int)
        st_frame = int(gt_keys[0])
        end_frame = int(gt_keys[-1])

        frame_ious = []
        for idx, pred_box in enumerate(pred_boxes):
            real_frame = st_frame + idx * frame_step
            frame_key = str(real_frame)

            if real_frame > end_frame:
                break

            if frame_key in gt_dict:
                gt_box = gt_dict[frame_key]
                frame_ious.append(iou(pred_box, gt_box))
            else:
                print(f"Missing GT for frame {frame_key}")

        mean_iou = np.mean(frame_ious) if frame_ious else 0.0
        all_video_ious.append(mean_iou)

        if mean_iou >= 0.3:
            vIoU_03 += 1
        if mean_iou >= 0.5:
            vIoU_05 += 1
        total_videos += 1

    mean_vIoU = np.mean(all_video_ious) if all_video_ious else 0.0
    vIoU_03_perc = (vIoU_03 / total_videos) * 100 if total_videos else 0
    vIoU_05_perc = (vIoU_05 / total_videos) * 100 if total_videos else 0

    return {
        "m_vIoU": round(mean_vIoU * 100, 2),
        "vIoU@0.3": round(vIoU_03_perc, 2),
        "vIoU@0.5": round(vIoU_05_perc, 2)
    }


def main():
    results = evaluate_llava_predictions(
        "/home/da530038/llava-grouding/LLaVA-Grounding/results/vidstg_llava_predictions.json")
    df = pd.DataFrame([results], index=["LLaVA Grounding"])
    print(df)


if __name__ == "__main__":
    main()
