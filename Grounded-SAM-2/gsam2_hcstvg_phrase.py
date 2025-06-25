import os
import json
import cv2
import torch
import numpy as np
import supervision as sv
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
#VIDEO_PATH = "./assets/hippopotamus.mp4"
#TEXT_PROMPT = "hippopotamus."
#OUTPUT_VIDEO_PATH = "./55_vfjywN5CN0Y__tracking_demo.mp4"
#SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
#SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# eval metric
def convert_to_x1y1x2y2(box):
    """
    Convert (x, y, w, h) format to (x1, y1, x2, y2).
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]

def iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Each box is in (x1, y1, x2, y2) format.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def best_iou(pred_boxes, gt_box):
    """
    Select the best matching IoU from multiple predicted boxes for a single ground truth box.
    """
    if len(pred_boxes) == 0:
        return 0  # No predictions, IoU is 0
    return max(iou(pred, gt_box) for pred in pred_boxes)

def box_area(box):
    """Calculate area of a box given as (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def biggest_iou(pred_boxes, gt_box):
    """
    Select the biggest bbox and calculate IoU.
    """
    if len(pred_boxes) == 0:
        return 0  # No predictions, IoU is 0
    # Find largest predicted box
    largest_pred_box = max(pred_boxes, key=box_area)

    return iou(largest_pred_box, gt_box)

def video_iou(pred_boxes_seq, gt_boxes_seq):
    """
    Compute video IoU (vIoU) by matching predicted boxes to ground truth frame by frame.
    :param pred_boxes_seq: List of lists of predicted bounding boxes per frame.
    :param gt_boxes_seq: List of ground truth bounding boxes per frame in (x, y, w, h) format.
    """
    #print("+++++++++++")
    #print("pred_boxes: ", pred_boxes_seq)
    #print("gt_boxes: ", gt_boxes_seq)
    #print("len pred: ", len(pred_boxes_seq))
    #print("len gt: ", len(gt_boxes_seq))

    assert len(pred_boxes_seq) == len(gt_boxes_seq), "Prediction and GT sequences must be the same length."
    
    gt_boxes_seq = [convert_to_x1y1x2y2(gt_box) for gt_box in gt_boxes_seq]  # Convert GT to (x1, y1, x2, y2)

    iou_scores = [biggest_iou(pred_boxes, gt_box) for pred_boxes, gt_box in zip(pred_boxes_seq, gt_boxes_seq)]
    #iou_scores = [best_iou(pred_boxes, gt_box) for pred_boxes, gt_box in zip(pred_boxes_seq, gt_boxes_seq)]
    
    return np.mean(iou_scores)

def process_IoU(predictions, ground_truths):
    """
    Compute vIoU for a single video.
    :param predictions: List of predicted bounding box sequences (multiple boxes per frame)
    :param ground_truths: List of ground truth bounding box sequences (one box per frame in (x, y, w, h) format)
    :return: vIoU
    """
    vIoU = video_iou(predictions, ground_truths)

    return vIoU

def process_video(VIDEO_PATH, TEXT_PROMPT, OUTPUT_VIDEO_PATH, SOURCE_VIDEO_FRAME_DIR, SAVE_TRACKING_RESULTS_DIR, gt_bbox, tube_start_frame, tube_end_frame):
    """
    Custom video input directly using video files
    """
    if not os.path.exists(SOURCE_VIDEO_FRAME_DIR):
        os.makedirs(SOURCE_VIDEO_FRAME_DIR)
        video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
        print(video_info)
        frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=15, start=tube_start_frame, end=tube_end_frame)

        # saving video to frames
        source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
    image = Image.open(img_path)
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]

    print(input_boxes)

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    OBJECTS = class_names

    print(OBJECTS)

    # prompt SAM 2 image predictor to get the mask for the object
    if input_boxes.size == 0:
        print("No objects detected, skipping this video.")
        print(f"The video is from {SOURCE_VIDEO_FRAME_DIR}")
        print(input_boxes)
        return 0  

    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """

    if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(SAVE_TRACKING_RESULTS_DIR)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    prediction=[]
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        
        prediction.append(detections.xyxy)
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # calculate vIoU
    vIoU = process_IoU(prediction, gt_bbox)
    print("vIoU: ", vIoU)

    """
    Step 6: Convert the annotated frames to video
    """
    create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
    return vIoU


"""
Read json files
"""
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_video_data(json_data, stop_num, version, g_cap):
    vIoU_list = []
    missing_key = []
    for video in json_data:
        trajectories = []
        video_ID = video['original_video_id']
        frame_count = video['frame_count']
        width = video['width']
        height = video['height']
        tube_start_frame = video['tube_start_frame']
        tube_end_frame = video['tube_end_frame']
        tube_start_time = video['tube_start_time']
        tube_end_time = video['tube_end_time']
        video_path = video['video_path']
        #caption = video['caption']
        if video_ID in g_cap:
            if version == 1:
                cap_data = g_cap[video_ID][0]
                caption = cap_data["phrases"][0]
            elif version == 2:
                caption = g_cap[video_ID]
                print("v2 referral caption: ", caption)
        else:
            print("===================")
            print("missing ", video_ID)
            print("===================")
            missing_key.append(video_ID)
            continue

        json_ID = video['video_id']
        if caption[len(caption)-1] != '.':
            caption = caption + '.'
        print("caption: ", caption)

        if tube_end_frame - frame_count >= (-1):
            tube_end_frame = frame_count-2
                        
        count = 0
        for point in video['trajectory']:
            if count%15 == 0 and count+tube_start_frame < tube_end_frame:
                trajectories.append(point)
            count+=1

        if version == 1:
            VIDEO_PATH = f"/home/c3-0/datasets/stvg/hcstvg1/v1/video/{video_path}"
            TEXT_PROMPT = caption
            OUTPUT_VIDEO_PATH = f"./hcstvgv1_eval_result/phrase/hcstvg_video_output/{video_ID}_tracking_demo.mp4"
            SOURCE_VIDEO_FRAME_DIR = f"./hcstvgv1_eval_result/phrase/hcstvg_video_frames/{video_ID}"
            SAVE_TRACKING_RESULTS_DIR = f"./hcstvgv1_eval_result/phrase/hcstvg_tracking_results/{video_ID}"
        elif version == 2:
            VIDEO_PATH = f"/home/c3-0/datasets/stvg/hcstvg2/videos/{video_path}"
            TEXT_PROMPT = caption
            OUTPUT_VIDEO_PATH = f"./hcstvgv2_eval_result/phrase/hcstvg_video_output/{video_ID}_tracking_demo.mp4"
            SOURCE_VIDEO_FRAME_DIR = f"./hcstvgv2_eval_result/phrase/hcstvg_video_frames/{video_ID}"
            SAVE_TRACKING_RESULTS_DIR = f"./hcstvgv2_eval_result/phrase/hcstvg_tracking_results/{video_ID}"

        if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
            os.makedirs(SAVE_TRACKING_RESULTS_DIR)
        
        vIoU = process_video(VIDEO_PATH, TEXT_PROMPT, OUTPUT_VIDEO_PATH, SOURCE_VIDEO_FRAME_DIR, SAVE_TRACKING_RESULTS_DIR, trajectories, tube_start_frame, tube_end_frame)
        vIoU_list.append(vIoU)
        
        #if json_ID == stop_num:
        #    break

    N = len(vIoU_list)
    if N == 0:
        return 0, 0, 0  # No videos

    m_vIoU = np.mean(vIoU_list)
    vIoU_3 = np.sum(np.array(vIoU_list) >= 0.3) / N
    vIoU_5 = np.sum(np.array(vIoU_list) >= 0.5) / N
    print("Total data number: ", N)

    print("Total missing keys:")
    for miss in missing_key:
        print(miss)

    return m_vIoU*100, vIoU_3*100, vIoU_5*100



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Grounded SAM2 Evaluation on VidSTG")
    parser.add_argument("--version", type=int, default=1)

    args = parser.parse_args()   

    version = args.version
    if version == 1:
        data = read_json_file("/home/c3-0/datasets/stvg/hcstvg1/test_proc.json")
        grounding_cap = read_json_file("/home/c3-0/datasets/stvg/preprocess_dump/hcstvg/hcstvg_pid_tubes_multi_sent_refined_v3/sentences_test.json")
    elif version == 2:
        data = read_json_file("/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json")
        grounding_cap = read_json_file("/home/we337236/stvg/dataset/hcstvg_v2/hcstvgv2_sentences_test_gpt_modified.json")

    num_vid = 4

    m_vIoU, vIoU_3, vIoU_5 = process_video_data(data, num_vid, version, grounding_cap)

    print('-'*50)
    print(f"Overall Results - m_vIoU: {m_vIoU:.4f}%, vIoU@3: {vIoU_3:.4f}%, vIoU@5: {vIoU_5:.4f}%")




