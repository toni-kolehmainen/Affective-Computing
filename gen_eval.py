import argparse
import os
from dataclasses import dataclass
from typing import Optional
import logging
import clip

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from rich import print as rprint
import pandas as pd

from src.models.base import EmotionCLIP
from src.datasets.genericvideodataset import GenericVideoDataset
from src.engine.utils import set_random_seed
from src.engine.logger import setup_logger
from config import settings

@dataclass
class EvalArgs(argparse.Namespace):
    ckpt_path: str = settings.EMOTIONCLIP_MODEL_PATH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    video_len: int = 8
    seed: int = 2022
    folder: Optional[str] = None

EMOTION_LIST = ["neutral", "joy", "sadness", "anger", "fear", "surprise"]

def DefaultDataLoader(dataset, args: EvalArgs):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )

@torch.no_grad()
def extract_features(model: EmotionCLIP, dataloader: DataLoader, args: EvalArgs):
    all_features = []
    all_targets = []
    clip_paths = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        frames, masks, targets, clip_ids, frame_names = batch

        frames = frames.to(args.device, non_blocking=True)
        masks = masks.to(args.device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
            feats = model.encode_video(frames, masks)
            feats = F.normalize(feats, dim=-1)

        all_features.append(feats.cpu())

        # Convert targets to tensor if available
        if targets is not None:
            targets_tensor = []
            for t in targets:
                if t is None:
                    targets_tensor.append(-1)
                else:
                    targets_tensor.append(t)
            all_targets.append(torch.tensor(targets_tensor))

        # Use folder + filename for zero-shot CSV
        clip_paths.extend([os.path.join(clip_id, fname) for clip_id, fname in zip(clip_ids, frame_names)])

    all_features = torch.cat(all_features).numpy()
    if all_targets:
        all_targets = torch.cat(all_targets).numpy()
        all_targets[all_targets == -1] = 0
    else:
        all_targets = None

    return all_features, all_targets, clip_paths

@torch.no_grad()
def zero_shot_emotion_analysis(model: EmotionCLIP, video_feats: np.ndarray, args: EvalArgs, clip_paths):
    rprint("\n[cyan]Running zero-shot emotion analysis...[/cyan]")

    prompts = [f"a person expressing {emo}" for emo in EMOTION_LIST]
    text_tokens = clip.tokenize(prompts).to(args.device)

    with torch.amp.autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
        text_feats = model.encode_text(text_tokens)
        text_feats = F.normalize(text_feats, dim=-1)

    video_feats = torch.from_numpy(video_feats).to(args.device)
    sims = video_feats @ text_feats.T
    preds = sims.argmax(dim=-1).cpu().numpy()
    pred_emotions = [EMOTION_LIST[i] for i in preds]

    folders = [os.path.dirname(p) for p in clip_paths]
    image_names = [os.path.basename(p) for p in clip_paths]

    results = pd.DataFrame({
        "folder": folders,
        "image_name": image_names,
        "predicted_emotion": pred_emotions
    })

    save_path = os.path.join(args.folder, "predicted_emotions.csv")
    results.to_csv(save_path, index=False)
    rprint(f"[green]✅ Saved zero-shot emotion predictions to:[/green] {save_path}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Folder containing subfolders with frames, optional human_boxes.json and answers.csv"
    )
    parser.add_argument("--ckpt-path", type=str, default="./data/emotionclip_latest.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    args_cli = parser.parse_args()

    args = EvalArgs(
        folder=args_cli.folder,
        ckpt_path=args_cli.ckpt_path,
        batch_size=args_cli.batch_size
    )

    set_random_seed(args.seed)
    logger = setup_logger("eval")
    rprint(args)

    # Load model
    model = EmotionCLIP(video_len=args.video_len, backbone_checkpoint=None)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(args.device)
    logger.info(f"Model loaded from {args.ckpt_path}")

    bbox_json = os.path.join(args.folder, "human_boxes.json") if os.path.exists(os.path.join(args.folder, "human_boxes.json")) else None
    answers_csv = os.path.join(args.folder, "answers.csv") if os.path.exists(os.path.join(args.folder, "answers.csv")) else None

    logger.info(f"Using folder: {args.folder}")
    if bbox_json: logger.info(f"Found bounding boxes: {bbox_json}")
    if answers_csv: logger.info(f"Found answers CSV: {answers_csv}")

    dataset = GenericVideoDataset(
        frames_root=args.folder,
        bbox_json=bbox_json,
        answers_csv=answers_csv,
        target="emotion_idx"
    )
    dataloader = DefaultDataLoader(dataset, args)

    X_test, y_test, clip_paths = extract_features(model, dataloader, args)

    if y_test is not None:
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 1:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X_test, y_test)
            preds = clf.predict(X_test)
            weighted_f1 = f1_score(y_test, preds, average="weighted") * 100
            acc = accuracy_score(y_test, preds) * 100
            logger.info("[Test] Weighted F1: %.2f | Accuracy: %.2f", weighted_f1, acc)
            print(f"\n✅ Evaluation complete:\nWeighted F1 = {weighted_f1:.2f}% | Accuracy = {acc:.2f}%")
        else:
            logger.warning("Only one class in data; skipping linear evaluation.")
            zero_shot_emotion_analysis(model, X_test, args, clip_paths)
    else:
        zero_shot_emotion_analysis(model, X_test, args, clip_paths)

if __name__ == "__main__":
    main()
