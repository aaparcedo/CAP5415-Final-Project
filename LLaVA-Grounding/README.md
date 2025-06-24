# LLaVA-Based Evaluation for Spatio-Temporal Grounding

This module implements a unified evaluation pipeline using the **LLaVA-Grounding model** for **referring expression grounding** across multiple video grounding benchmarks.

It is part of a general framework for benchmarking **vision-language models** (VLMs) on tasks that require **spatio-temporal understanding**, such as:

- HC-STVG v1
- HC-STVG v2
- VidSTG
- Something-Else
- VidVRD

---

## Evaluation Overview

LLaVA grounding combines **visual and language processing** through a conversational multi-modal architecture. This script:

- Loads a frozen LLaVA-Grounding model (interleaved vision-language)
- Processes a temporal segment (tube) of sampled video frames
- Uses **caption-guided inference** to produce bounding boxes
- Evaluates the predictions using:
  - **mvIoU**: Mean IoU between predicted and ground truth boxes
  - **mtIoU**: Temporal tube alignment score
- Saves per-entry results to structured `.json`

---

## Evaluation Flow

```text
1. Load model checkpoints + configuration files
2. Load dataset-specific video and annotation structure
3. For each video entry:
    ├── Extract tube frames [start, end]
    ├── Sample every Nth frame (e.g., every 15th frame)
    ├── For each frame:
    │   ├── Run caption-grounded inference using LLaVA
    │   └── **Take the first predicted bounding box**
    ├── Convert normalized boxes to [x, y, w, h]
    ├── Compute:
    │   ├── mvIoU (mean IoU across sampled frames)
    │   └── mtIoU (temporal intersection over union)
4. Save predictions to JSON

