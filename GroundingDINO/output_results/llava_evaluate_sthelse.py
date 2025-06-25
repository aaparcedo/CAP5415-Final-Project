import json
import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """IoU between [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def evaluate_llava_sthelse(json_path, frame_step=15):
    with open(json_path, "r") as f:
        data = json.load(f)

    mv_ious = []
    vIoU_03, vIoU_05 = 0, 0

    for sample in data:
        pred_boxes = sample.get("predicted_boxes", [])
        gt_boxes = sample.get("ground_truth_boxes", {})

        if not pred_boxes or not gt_boxes:
            continue

        sorted_gt_frame_ids = sorted([int(fid) for fid in gt_boxes.keys()])
        if not sorted_gt_frame_ids:
            continue

        st_frame = sorted_gt_frame_ids[0]
        frame_ids = [st_frame + i * frame_step for i in range(len(pred_boxes))]

        frame_ious = []
        for pred_box, fid in zip(pred_boxes, frame_ids):
            fid_str = str(fid)
            if fid_str in gt_boxes:
                gt_box = gt_boxes[fid_str]
                iou = calculate_iou(pred_box, gt_box)
                frame_ious.append(iou)

        if frame_ious:
            mv_iou = np.mean(frame_ious)
            mv_ious.append(mv_iou)
            if mv_iou >= 0.3:
                vIoU_03 += 1
            if mv_iou >= 0.5:
                vIoU_05 += 1

    n = len(mv_ious)
    return {
        "m_vIoU": round(np.mean(mv_ious) * 100, 2) if n else 0,
        "vIoU@0.3": round(vIoU_03 / n * 100, 2) if n else 0,
        "vIoU@0.5": round(vIoU_05 / n * 100, 2) if n else 0,
        "Num samples": n
    }


def main():
    result_path = "/home/da530038/llava-grouding/LLaVA-Grounding/output_results/somethingelse_llava_predictions.json"
    results = evaluate_llava_sthelse(result_path, frame_step=15)
    df = pd.DataFrame([results], index=["LLaVA-Grounding"])
    print(df)


if __name__ == "__main__":
    main()
