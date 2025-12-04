# VISTA: Video Interaction Spatio-Temporal Analysis Benchmark

This repository contains the evaluation code for **VISTA**, a benchmark for evaluating fine-grained, interaction-centric spatio-temporal reasoning in Vision-Language Models (VLMs).

## Overview

VISTA evaluates VLMs on their ability to perform spatio-temporal grounding—localizing subjects mentioned in natural language queries across video frames. The benchmark supports evaluation on multiple datasets and three specialist MLLMs:

- **CogVLM** (Grounding version)
- **Shikra**
- **Ferret-v1**

## Project Structure

```
CAP5415-Final-Project/
├── VISTA/
│   ├── __init__.py          # Package initialization
│   ├── datasets.py          # Dataset loaders for all supported datasets
│   ├── utils.py             # Utility functions (IoU calculation, box conversion, etc.)
│   └── models.py            # Model wrappers for CogVLM, Shikra, and Ferret
├── shikra/                  # Shikra model submodule (don't forget to clone)
├── ml-ferret/               # Ferret model submodule (don't forget to clone)
├── run_eval.py              # Main evaluation script
└── README.md
```

## Requirements

### Dependencies

```bash
# Create conda environment
conda create -n stvg python=3.10
conda activate stvg

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install transformers
pip install opencv-python
pip install numpy
pip install tqdm
pip install pillow
pip install pycocotools
pip install mmengine
pip install bitsandbytes

# For Shikra
cd shikra
pip install -e .

# For Ferret
cd ml-ferret
pip install -e .
```

### Model Weights

Download the pretrained model weights:

- **CogVLM**: Automatically downloaded from `zai-org/cogvlm-grounding-generalist-hf`
- **Shikra**: Download from [shikras/shikra-7b](https://huggingface.co/shikras/shikra-7b) and place in `/CAP5415-Final-Project/shikra/shikras/shikra-7b`
  - Model code: [https://github.com/shikras/shikra/tree/main/mllm](https://github.com/shikras/shikra/tree/main/mllm)
- **Ferret**: Download from Apple's ML-Ferret repository and place in `/CAP5415-Final-Project/ml_ferret/ferret-7b-v1-3`
  - Model code: [https://github.com/apple/ml-ferret/tree/main/ferret](https://github.com/apple/ml-ferret/tree/main/ferret)

**Note**: Update the paths in `models.py` to match your local setup.

## Supported Datasets

| Dataset | Task Types | Description |
|---------|------------|-------------|
| HC-STVG v1 | referral, freeform | Human-centric spatio-temporal video grounding |
| HC-STVG v2 | referral, freeform | Extended version with more videos |
| VidSTG | referral, freeform | Video spatio-temporal grounding |
| VidVRD | referral, freeform | Video visual relation detection |
| MeViS | freeform | Motion expressions video segmentation |
| RVOS | freeform | Referring YouTube-VOS |

**Note**: Update dataset paths in `VISTA/datasets.py` to point to your local data directories.

## Usage

### Basic Evaluation

```bash
python run_eval.py \
    --dataset <dataset_name> \
    --model <model_name> \
    --task_type <task_type> \
    --output_path <output_json_path> \
    --device cuda
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset` | Yes | Dataset name: `hcstvg1`, `hcstvg2`, `vidstg`, `vidvrd`, `mevis`, `rvos` |
| `--model` | Yes | Model name: `cogvlm`, `shikra`, `ferret` |
| `--task_type` | No* | Task type: `referral` or `freeform` (*required for hcstvg, vidstg, vidvrd) |
| `--output_path` | Yes | Path for output JSON file |
| `--device` | No | Device for inference (default: `cuda`) |
| `--frame_step` | No | Frame sampling interval (default: `5`) |
| `--entry_index` | No | Starting index for evaluation (default: `-1` for full dataset) |
| `--max_iters` | No | Maximum samples to process (default: `-1` for all) |

### Examples

**Evaluate CogVLM on HC-STVG v1 with freeform queries:**
```bash
python run_eval.py \
    --dataset hcstvg1 \
    --model cogvlm \
    --task_type freeform \
    --output_path results/cogvlm_hcstvg1_freeform.json \
    --device cuda
```

**Evaluate Shikra on MeViS:**
```bash
python run_eval.py \
    --dataset mevis \
    --model shikra \
    --output_path results/shikra_mevis.json \
    --device cuda
```

**Debug mode (process only 2 samples):**
```bash
python run_eval.py \
    --dataset vidstg \
    --model ferret \
    --task_type referral \
    --output_path results/debug.json \
    --entry_index 0 \
    --max_iters 2
```

## Output Format

The evaluation script outputs a JSON file with the following structure:

```json
{
    "evaluation_parameters": {
        "frame_step": 5,
        "dataset": "hcstvg1",
        "model": "cogvlm",
        "task_type": "freeform"
    },
    "timing_summary": {
        "total_evaluation_time_seconds": 1234.56,
        "total_model_inference_time_seconds": 1000.00,
        "total_samples_processed": 100,
        "total_frames_processed": 500
    },
    "overall_results": {
        "avg_mviou": 0.5470,
        "avg_mviou03": 0.65,
        "avg_mviou05": 0.55
    },
    "results": [...]
}
```

## Evaluation Metrics

- **m_vIoU (Mean Video IoU)**: Average spatio-temporal IoU across all frames
- **m_vIoU@0.3**: Percentage of frames with IoU ≥ 0.3
- **m_vIoU@0.5**: Percentage of frames with IoU ≥ 0.5


## Acknowledgments

This project uses the following open-source models:
- [CogVLM](https://github.com/THUDM/CogVLM)
- [Shikra](https://github.com/shikras/shikra) - specifically [mllm module](https://github.com/shikras/shikra/tree/main/mllm)
- [Ferret](https://github.com/apple/ml-ferret) - specifically [ferret module](https://github.com/apple/ml-ferret/tree/main/ferret)

## License

This project is for academic use only.