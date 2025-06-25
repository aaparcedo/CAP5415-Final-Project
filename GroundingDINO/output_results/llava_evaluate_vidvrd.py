import json
import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """IoU for [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def evaluate_llava_vidvrd(json_path, frame_step=15):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_mv_ious = []
    vIoU_03 = vIoU_05 = 0

    for entry in data:
        pred_boxes = entry.get("predicted_boxes", [])
        gt_dict = entry.get("ground_truth_boxes", {})
        frame_ids = entry.get("frame_ids", [])

        if not pred_boxes or not gt_dict or not frame_ids:
            continue

        if len(pred_boxes) != len(frame_ids):
            print(f"[Warning] Mismatched length in {entry.get('video_path')}")
            continue

        frame_ious = []
        for pred_box, frame_id in zip(pred_boxes, frame_ids):
            if (frame_id - frame_ids[0]) % frame_step != 0:
                continue  # ignore frames not part of sampling
            fid_str = str(frame_id)
            if fid_str in gt_dict:
                gt_box = gt_dict[fid_str]
                iou_val = calculate_iou(pred_box, gt_box)
                frame_ious.append(iou_val)

        if frame_ious:
            mv_iou = np.mean(frame_ious)
            all_mv_ious.append(mv_iou)
            if mv_iou >= 0.3:
                vIoU_03 += 1
            if mv_iou >= 0.5:
                vIoU_05 += 1

    n = len(all_mv_ious)
    return {
        "m_vIoU": round(np.mean(all_mv_ious) * 100, 2) if n else 0,
        "vIoU@0.3": round(vIoU_03 / n * 100, 2) if n else 0,
        "vIoU@0.5": round(vIoU_05 / n * 100, 2) if n else 0,
        "Num samples": n
    }


def main():
    result_path = "/home/da530038/llava-grouding/LLaVA-Grounding/results/vidvrd_llava_predictions.json"
    frame_step = 15
    results = evaluate_llava_vidvrd(result_path, frame_step=frame_step)
    df = pd.DataFrame([results], index=["LLaVA-Grounding"])
    print(df)


if __name__ == "__main__":
    main()
