from __future__ import annotations
import os
import os.path as osp
from typing import Literal, Optional
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import orjson

from .utils import get_correct_bbox, bbox_to_mask
from ..models.tokenizer import tokenize
from config import settings

EMOTION_CLASS_NAMES = [
    'neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'
]

SENTIMENT_CLASS_NAMES = [
    'neutral', 'positive', 'negative'
]

logger = logging.getLogger()

class MELD(Dataset):
    def __init__(
        self,
        data_dir: str = settings.MELD_DATASET_PATH,
        split: Literal['train', 'dev', 'test'] = 'test',
        video_len: int = 8,
        target: Literal['utt_text', 'utt_token', 'emotion_idx', 'sentiment_idx', 'multimodal'] = 'emotion_idx',
        frames_root: Optional[str] = None,
        bbox_json: Optional[str] = None,
        csv_file: Optional[str] = None,
    ):
        assert split in ['train', 'dev', 'test']
        super().__init__()

        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        self.BBOX_TO_MASK_THRESHOLD = 0.5

        self.data_dir = data_dir
        self.split = split
        self.video_len = video_len
        self.target = target
        self.frames_root = frames_root
        self.bbox_json = bbox_json
        self.csv_file = csv_file

        self._create_index()
        self.preprocesser = T.Compose([
            T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _create_index(self):
        # Load CSV
        annotation_file_path = self.csv_file if self.csv_file is not None else osp.join(self.data_dir, 'MELD.Raw', f'{self.split}_sent_emo.csv')
        self.index = pd.read_csv(annotation_file_path)

        # Preprocessing
        self.index['Emotion'] = self.index['Emotion'].apply(lambda x: EMOTION_CLASS_NAMES.index(x))
        self.index['Sentiment'] = self.index['Sentiment'].apply(lambda x: SENTIMENT_CLASS_NAMES.index(x))

        # Load bounding boxes
        boxes_fpath = self.bbox_json if self.bbox_json is not None else osp.join(self.data_dir, f'{self.split}_human_boxes.json')
        logger.info(f"Loaded bounding boxes from {boxes_fpath}")
        with open(boxes_fpath, 'r') as f:
            self.human_boxes = orjson.loads(f.read())

        # Filter CSV to only clips that exist in frames_root
        if self.frames_root is not None:
            existing_clips = set(os.listdir(self.frames_root))
            self.index['clip_id'] = self.index.apply(lambda row: f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}_frames", axis=1)
            self.index = self.index[self.index['clip_id'].isin(existing_clips)].reset_index(drop=True)

        logger.info(f'Index of {self.split} set created, {self.index.shape[0]} samples in total.')

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, i):
        clip_id = self.index.loc[i, 'clip_id']
        clip_dir = osp.join(self.frames_root, clip_id) if self.frames_root else osp.join(self.data_dir, 'frames', f'{self.split}_splits', clip_id, 'frames')

        frame_files = sorted(os.listdir(clip_dir))
        num_frames = len(frame_files)
        if num_frames == 0:
            raise RuntimeError(f"No frames found in {clip_dir}")

        # Interpolate frame indices to always get video_len frames
        sampled_frame_ids = np.linspace(0, num_frames-1, self.video_len, dtype=int)

        frames, masks = [], []
        for idx in sampled_frame_ids:
            frame_name = frame_files[idx]
            frame_path = osp.join(clip_dir, frame_name)
            raw_frame = Image.open(frame_path).convert('RGB')

            raw_boxes = self.human_boxes.get(clip_id, {}).get(frame_name, [])

            resized_frame = F.resize(raw_frame, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
            resized_boxes = get_correct_bbox(raw_boxes, resized_frame.size)
            mask = bbox_to_mask(resized_boxes, resized_frame.size, binary_threshold=self.BBOX_TO_MASK_THRESHOLD)

            frame = F.center_crop(resized_frame, self.CROP_SIZE)
            mask = F.center_crop(mask, self.CROP_SIZE)
            frame = F.normalize(F.to_tensor(frame), mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

            frames.append(frame)
            masks.append(mask)

        frames = torch.stack(frames, dim=0).float()
        video_masks = torch.stack(masks, dim=0).float()

        # Text/target
        if self.target == 'utt_text':
            target = self.index.loc[i, 'Utterance']
        elif self.target == 'utt_token':
            target = tokenize(self.index.loc[i, 'Utterance']).squeeze()
        elif self.target == 'emotion_idx':
            target = self.index.loc[i, 'Emotion']
        elif self.target == 'sentiment_idx':
            target = self.index.loc[i, 'Sentiment']
        elif self.target == 'multimodal':
            target = {
                'utt_text': self.index.loc[i, 'Utterance'],
                'utt_token': tokenize(self.index.loc[i, 'Utterance']).squeeze(),
                'emotion_idx': self.index.loc[i, 'Emotion'],
            }
        else:
            raise NotImplementedError

        return frames, video_masks, target

if __name__ == '__main__':
    pass
