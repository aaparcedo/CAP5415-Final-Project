import argparse
import os
import sys

import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    print("+++++++++++")
    #print("pred_boxes: ", pred_boxes_seq)
    #print("gt_boxes: ", gt_boxes_seq)
    print("len pred: ", len(pred_boxes_seq))
    print("len gt: ", len(gt_boxes_seq))

    #assert len(pred_boxes_seq) <= len(gt_boxes_seq), "Prediction and GT sequences must be the same length."

    iou_scores = []
    for i, gt_box in enumerate(gt_boxes_seq):
        gt_box_converted = convert_to_x1y1x2y2(gt_box)

        if i < len(pred_boxes_seq):
            pred_boxes = pred_boxes_seq[i]
            iou = biggest_iou(pred_boxes, gt_box_converted)
        else:
            # No prediction available for this frame
            iou = 0.0

        iou_scores.append(iou)

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

def run_coco_eval_per_image_boxes(pred_boxes, gt_boxes, image_size=(640, 480), category_name="object"):
    """
    Run COCOeval where each bbox (GT and prediction) belongs to a different image.

    Args:
        pred_boxes (list): List of predicted bboxes [x, y, w, h] (one per image), no score.
        gt_boxes (list): List of ground truth bboxes [x, y, w, h] (one per image).
        image_size (tuple): Tuple (width, height) for all images.
        category_name (str): Category name.

    Returns:
        float: mAP at IoU 0.5:0.95
    """

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    print("--------Before----------")
    print("pred_boxes len: ", num_pred)
    print("gt_boxes len: ", num_gt)

    # Trim extra preds or pad with dummy boxes
    if num_pred > num_gt:
        pred_boxes = pred_boxes[:num_gt]
    elif num_pred < num_gt:
        # Add dummy boxes outside image to force IoU = 0
        dummy_box = [image_size[0] + 1000, image_size[1] + 1000, 10, 10]
        pred_boxes += [[dummy_box]] * (num_gt - num_pred)

    print("-------After---------")
    print("pred_boxes len: ", len(pred_boxes))
    print("gt_boxes len: ", len(gt_boxes))

    assert len(pred_boxes) == len(gt_boxes), "Mismatch in number of predicted and GT boxes"

    category_info = [{"id": 1, "name": category_name}]
    image_info = []
    gt_annotations = []
    pred_annotations = []

    #print("------------")
    #print("pred boxes len: ", len(pred_boxes))
    #print("pred boxes: ", pred_boxes)
    for i, (gt_box, pred_box) in enumerate(zip(gt_boxes, pred_boxes)):
        image_id = int(i + 1)

        image_info.append({
            "id": image_id,
            "width": int(image_size[0]),
            "height": int(image_size[1])
        })

        x, y, w, h = [float(v) for v in gt_box]
        gt_annotations.append({
            "id": int(i),
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": float(w * h),
            "iscrowd": 0
        })
        print("+++++++")
        print("pred box: ", pred_box)
        if len(pred_box)==0:
            x=0
            y=0
            w=0
            h=0
        else:
            x, y, w, h = [float(v) for v in pred_box[0]]
        pred_annotations.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "score": 1.0  # default dummy score
        })

    gt_dict = {
        "images": image_info,
        "annotations": gt_annotations,
        "categories": category_info
    }

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as gt_file, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as pred_file:

        json.dump(gt_dict, gt_file)
        gt_file.flush()

        json.dump(pred_annotations, pred_file)
        pred_file.flush()

        coco_gt = COCO(gt_file.name)
        coco_dt = coco_gt.loadRes(pred_file.name)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # AP at IoU 0.5:0.95 is stored in stats[0]
        return coco_eval.stats[0]


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

class sth_else_Dataset:
    def __init__(self, caption_type, transform=None):
        if caption_type == "free_form":
            ann_path = ""
            frame_path = ""
        elif caption_type =="referral":
            ann_path = "/home/we337236/stvg/dataset/something_else/something_else_vidstg_format.json"
            frame_path = "/home/we337236/stvg/Grounded-SAM-2/sth_else_eval_result/referral/video_frames"
        elif caption_type == "phrase":
            ann_path = "/home/we337236/stvg/dataset/something_else/something_else_coco_format.json"
            frame_path = "/home/we337236/stvg/Grounded-SAM-2/sth_else_eval_result/phrase/video_frames"

        self.data = read_json_file(ann_path)  # Ensure it's a list
        self.transform = transform
        self.video_frame_path = frame_path
        self.caption_type = caption_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        frame_root = self.video_frame_path

        bbox = []
        tube_start_frame = video['st_frame']
        tube_end_frame = video['ed_frame']

        if tube_start_frame >= tube_end_frame:
            print("hand and obj did not overlap skip the video")
            return -1, -1
        elif tube_end_frame - tube_start_frame < 15:
            print("video is shorter than 15 frames")
            return -1, -1

        video_path = video['video_path']
        #detect_type = video['type']
        #question_type = video['qtype']
        #tID = video['target_id']
        vID = os.path.basename(video_path).split('.')[0]
        #ID = video["ID"]
        video_ID = f"{vID}_{tube_start_frame}_{tube_end_frame}"
                
        caption = video['caption']
        print("caption: ", caption)
        if caption[len(caption)-1] != '.':
            caption = caption + '.'
        
        count = 0 
        for key, value in video["bbox"].items():
            if count%15 == 0 and count+tube_start_frame < tube_end_frame:
                bbox.append(value)
            count+=1

        frame_list = []
        frame_path = f"{frame_root}/{video_ID}"
        for filename in os.listdir(frame_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                img_path = os.path.join(frame_path, filename)
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image, _ = transform(image, None)  # 3, h, w
                frame_list.append(image)
        if not frame_list:
            return -1, -1
        
        if self.caption_type == "phrase":
            entry = {
                "caption": caption,
                "category": video['category'],
                "bbox": bbox,
                "width": video['width'],
                "height": video['height']
            }
        else:
            entry = {
                "caption": caption,
                "bbox": bbox,
                "width": video['width'],
                "height": video['height']
            }

        return entry, frame_list

def run_grounded_sam(model, image, image_height, image_width, text_prompt, box_threshold, text_threshold, device):

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    if boxes_filt.nelement() == 0:
        return boxes_filt,0,0

    predictor.set_image(image)

    H, W = image_height, image_width
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    return boxes_filt, masks, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("gsam eval on hcstvg", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    #parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    #parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    
    parser.add_argument("--dataset_version", type=int, default=1)
    parser.add_argument("--caption_type", type=str, default="referral")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    #image_path = args.input_image
    #text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = sth_else_Dataset(args.caption_type, transform)

    data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x))
    )

    vIoU_list=[]
    count = 0
    for entries, frame_lists in tqdm(data_loader, desc=f"Evaluating something_else"):
        for entry, frame_list in zip(entries, frame_lists):
            if entry == -1:
                continue
            image_height = entry["height"]
            image_width = entry["width"]
            text_prompt = entry["caption"]
            if text_prompt == ".":
                continue
            gt_bbox = entry["bbox"]
            prediction = []
            for image in frame_list:
                boxes_filt, masks, pred_phrases = run_grounded_sam(model, image, image_height, image_width, text_prompt, box_threshold, text_threshold, device)
                prediction.append(boxes_filt)
            # calculate vIoU
            if args.caption_type == "phrase":
                vIoU = run_coco_eval_per_image_boxes(prediction, gt_bbox, image_size=(image_width, image_height), category_name=entry["category"])
            else:
                vIoU = process_IoU(prediction, gt_bbox)
            print("vIoU: ", vIoU)
            vIoU_list.append(vIoU)
        count += 1
        #if count == 4:
        #    break
    
    N = len(vIoU_list)

    m_vIoU = np.mean(vIoU_list)
    vIoU_3 = np.sum(np.array(vIoU_list) >= 0.3) / N
    vIoU_5 = np.sum(np.array(vIoU_list) >= 0.5) / N
    print("Total data number: ", N)
    
    m_vIoU = m_vIoU * 100
    vIoU_3 = vIoU_3 * 100
    vIoU_5 = vIoU_5 * 100

    print('-'*50)
    print(f"Overall Results - m_vIoU: {m_vIoU:.4f}%, vIoU@3: {vIoU_3:.4f}%, vIoU@5: {vIoU_5:.4f}%")




