import argparse
import time
from pathlib import Path
import os
import sys

yolov7_path = Path(__file__).resolve().parents[2] / "yolov7"
sys.path.insert(0, str(yolov7_path))

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from datasets import ImagesDataset
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    all_human_boxes = dict()

    t0 = time.time()
    for i, vid in enumerate(tqdm(os.listdir(source))):
    # for vid in tqdm(os.listdir(source)):

        # Tonttu test
        # if i >= 3:
        #     break

        clip_boxes = detect_one_video(model, source, vid, save_dir, stride, device, half)
        all_human_boxes.update(clip_boxes)
        # detect_one_video(model,source,vid,save_dir,stride,device,half,webcam=False)

    with open(os.path.join(source, 'test_bounding_boxes.json'), 'w') as f:
        json.dump(all_human_boxes, f, indent=2)

    print(f'Done. ({time.time() - t0:.3f}s)')

def detect_one_video(model, source, vid, save_dir, stride, device, half, webcam=False):
    save_dir = save_dir / vid
    frames_folder = os.path.join(source, vid, 'frames')
    human_boxes_path = os.path.join(source, f'{vid}_human_boxes.json')

    if os.path.exists(human_boxes_path):
        return True
    if not os.path.exists(frames_folder):
        raise FileNotFoundError(f"{frames_folder} does not exist")

    # Dataloader
    if webcam:
        dataset = LoadStreams(frames_folder, img_size=opt.img_size, stride=stride)
    else:
        dataset = ImagesDataset(frames_folder, img_size=opt.img_size, stride=stride)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))

    clip_human_boxes = dict()  # per video

    for path, img, im0s in dataset:
        img = img.to(device).half() if device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms, multi_label=True)

        for i, det in enumerate(pred):
            frame_name = Path(path[i]).name  # 00000001.jpg
            boxes = det[det[:, -1] == 0, :4].cpu().detach().numpy().tolist()  # only class 0 (person)
            clip_human_boxes[frame_name] = boxes
            boxes_with_conf_class = []
            for *xyxy, conf, cls in reversed(det): # unpack box + confidence + class
                if int(cls) == 0: # only class 0 (person)
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    boxes_with_conf_class.append(xywh + [float(conf), int(cls)])
                    clip_human_boxes[frame_name] = boxes_with_conf_class

    # Save JSON using the video/clip name as key
    return {vid: clip_human_boxes}

    # human_boxes = {vid: clip_human_boxes}
    # with open(human_boxes_path, 'w') as f:
    #     json.dump(human_boxes, f, indent=2)

    # return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../youtube', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=32, help='inference batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='inference num workers')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
