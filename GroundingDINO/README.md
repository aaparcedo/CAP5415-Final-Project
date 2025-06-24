#  GroundingDINO-Based Evaluation Suite for Video Grounding Benchmarks

This repository provides a modular evaluation framework built on **GroundingDINO** to benchmark multiple **spatio-temporal video grounding datasets**, such as:

- HC-STVG v1
- HC-STVG v2
- VidSTG
- Something-Else
- VidVRD

Each dataset-specific evaluation script follows a **shared pipeline design** to ensure consistent processing and fair comparison.

---

## General Evaluation Flow

```text
1. Load GroundingDINO model (config + checkpoint)
2. Load dataset-specific JSON annotations + videos
3. For each video entry:
    ├── Extract relevant tube/frame segment
    ├── Apply image transforms and preprocessing
    ├── Predict bounding boxes guided by textual captions
    ├── **Select most confident box** (per frame)
    └── Evaluate:
        ├── mvIoU – Mean IoU across predicted vs ground truth boxes
        └── mtIoU – Temporal IoU for tube alignment
4. Save per-entry results to JSON
5. Print dataset-level evaluation metrics
