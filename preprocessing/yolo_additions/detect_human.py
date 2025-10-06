import argparse
import time
from pathlib import Path
import os
import orjson

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, ImagesDataset
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source = Path(opt.source)
    weights = opt.weights

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    opt.img_size = check_img_size(opt.img_size, s=stride)

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()

    all_human_boxes = dict()
    t0 = time.time()

    for video_dir in tqdm(sorted(source.iterdir())):
        if not video_dir.is_dir() or not video_dir.name.startswith("dia"):
            continue

        clip_id = video_dir.name
        video_boxes = detect_one_video(
            model=model,
            source_dir=video_dir,
            device=device,
            half=half,
            stride=stride,
        )
        if video_boxes:
            all_human_boxes[clip_id] = video_boxes

    output_json = source.parent / "human_boxes.json"
    with open(output_json, 'wb') as f:
        f.write(orjson.dumps(all_human_boxes))

    print(f"Saved human boxes to: {output_json}")
    print(f"Done. ({time.time() - t0:.3f}s)")



def detect_one_video(model, source_dir, device, half, stride):
    dataset = ImagesDataset(str(source_dir), img_size=opt.img_size, stride=stride)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

    names = model.module.names if hasattr(model, 'module') else model.names
    human_boxes = dict()

    model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))  # warmup

    for path, img, im0s in dataloader:
        img = img.to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, multi_label=True)

        for i, det in enumerate(pred):
            p = Path(path[i])
            frame_name = p.name
            if det is not None and len(det):
                human_det = det[:,:-1][det[:,-1]==0]  # class 0 = person
                if len(human_det):
                    human_boxes[frame_name] = human_det.cpu().numpy().tolist()
                else:
                    human_boxes[frame_name] = []
            else:
                human_boxes[frame_name] = []

    return human_boxes if len(human_boxes) else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, required=True, help='Path to folder containing *_frames folders')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=32, help='inference batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='inference num workers')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug', action='store_true', help='debug mode, i.e. example image with bboxes')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
