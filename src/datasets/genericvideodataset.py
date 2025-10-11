import os
import os.path as osp
from typing import Optional, Literal
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import json

from .utils import get_correct_bbox, bbox_to_mask
from ..models.tokenizer import tokenize

logger = logging.getLogger()

class GenericVideoDataset(Dataset):
    """
    Generic dataset where each image is treated as an independent sample.
    Can handle optional bounding boxes and answer labels.
    """

    def __init__(
        self,
        frames_root: str,
        bbox_json: Optional[str] = None,
        answers_csv: Optional[str] = None,
        target: Literal['emotion_idx', 'utt_text', 'utt_token'] = 'emotion_idx',
        preprocesser: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        self.frames_root = frames_root
        self.target = target

        # Load bounding boxes if provided
        if bbox_json and osp.exists(bbox_json):
            with open(bbox_json, 'r') as f:
                self.human_boxes = json.load(f)
        else:
            self.human_boxes = {}

        # Load answers CSV if exists
        if answers_csv and osp.exists(answers_csv):
            self.index = pd.read_csv(answers_csv)
            if 'emotion' in self.index.columns:
                self.emotion_classes = sorted(self.index['emotion'].unique())
                self.index['Emotion'] = self.index['emotion'].apply(lambda x: self.emotion_classes.index(x))
            else:
                self.emotion_classes = []
        else:
            self.index = pd.DataFrame(columns=['clip_id', 'frame_name', 'Emotion'])
            self.emotion_classes = []

        # Collect all image files in frames_root
        image_rows = []
        for subdir in os.listdir(frames_root):
            subdir_path = osp.join(frames_root, subdir)
            if osp.isdir(subdir_path):
                for fname in sorted(os.listdir(subdir_path)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        row = {'clip_id': subdir, 'frame_name': fname}
                        if not self.index.empty:
                            match = self.index[(self.index['clip_id'] == subdir) & 
                                               (self.index['frame_name'] == fname)]
                            if not match.empty:
                                row['Emotion'] = int(match.iloc[0]['Emotion'])
                        image_rows.append(row)
        self.index = pd.DataFrame(image_rows)
        logger.info(f"Loaded {len(self.index)} images from {frames_root}")

        # Preprocessing
        self.preprocesser = preprocesser or T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        clip_id = row['clip_id']
        frame_name = row['frame_name']
        frame_path = osp.join(self.frames_root, clip_id, frame_name)
        frame = Image.open(frame_path).convert('RGB')

        # Apply bounding box mask if exists
        raw_boxes = self.human_boxes.get(clip_id, {}).get(frame_name, [])
        mask = bbox_to_mask(raw_boxes, frame.size) if raw_boxes else torch.ones(1, frame.height, frame.width)

        # Preprocess
        frame = self.preprocesser(frame)
        mask = F.resize(mask, [224, 224]).float()

        # Load target if exists
        if self.target == 'emotion_idx' and 'Emotion' in row:
            target = row['Emotion']
        else:
            target = -1

        # Return clip_id and frame_name along with data
        return frame.unsqueeze(0), mask.unsqueeze(0), target, clip_id, frame_name

