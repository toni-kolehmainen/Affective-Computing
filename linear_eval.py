import argparse
import os 
import os.path as osp
from dataclasses import dataclass
import logging
from typing import Literal, Optional

from matplotlib import transforms
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, r2_score
from scipy.signal import savgol_filter
from rich import print as rprint

from src.models.base import EmotionCLIP
from src.datasets.meld import MELD
from src.datasets.bold import BoLD
from src.datasets.movie_graphs import MovieGraphsDataset
from src.datasets.emotic import Emotic
from src.datasets.custom_image_dataset import CustomImageDataset
from src.datasets.liris_accede import LirisAccede
from src.engine.utils import set_random_seed
from src.engine.logger import setup_logger


@dataclass
class EvalArgs(argparse.Namespace):
    dataset: Literal['bold', 'mg', 'meld', 'emotic', 'la'] = 'bold'
    ckpt_path: str = './exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt'
    use_cache: bool = False # when use_cache=True, ckpt_path and save_path are ignored
    save_cache: bool = True
    
    ckpt_strict: bool = True
    cuda_deterministic: bool = True
    # cuda_deterministic: bool = False
    has_test_set: bool = False
    seed: int = 2022
    video_len: int = 8
    device: str = 'cuda:0'
    dataloader_workers: int = 4
    batch_size: int = 128
    cache_dir: str = './data/cache'


@torch.no_grad()
def extract_features(
    model: EmotionCLIP, 
    dataloader: DataLoader, 
    args: EvalArgs,
    split: str,
    data_type: str = 'video',
):
    feature_extractor = model.encode_video if data_type == 'video' else model.encode_image
    tdataloader = tqdm(dataloader, desc=f'Extracting features ({split})', unit_scale=dataloader.batch_size)
    all_features = []
    all_targets = []
    for i, batch in enumerate(tdataloader):
        # load batch
        visual_inputs, visual_masks, targets = batch
        visual_inputs = visual_inputs.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        visual_masks = visual_masks.to(args.device, non_blocking=True)
        # forward
        with torch.cuda.amp.autocast():
            features = feature_extractor(visual_inputs, visual_masks)
            features = F.normalize(features, dim=-1)
        all_features.append(features)
        all_targets.append(targets)
    all_features = torch.cat(all_features).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    return all_features, all_targets


def DefaultDataLoader(dataset: Dataset, args: EvalArgs):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_workers,
        pin_memory=True
    )

@torch.no_grad()
def accuracy_from_affect2mm(target: np.ndarray, output: np.ndarray):
    output = torch.from_numpy(output)
    target = torch.from_numpy(target)
    _, pred = output.topk(1, 1, True, True)
    target_value = torch.gather(target, 1, pred)
    correct_k = (target_value > 0).float().sum(0, keepdim=False).sum(0, keepdim=True)
    correct_k /= target.shape[0]
    res = correct_k.mul_(100.0).item()
    return res


def eval_liris_accede(model:EmotionCLIP, args: EvalArgs):
    train_dataset = LirisAccede(split='train')
    test_dataset = LirisAccede(split='test')
    train_dataloader = DefaultDataLoader(train_dataset, args)
    test_dataloader = DefaultDataLoader(test_dataset, args)
    X_train, y_train = extract_features(model, train_dataloader, args, split='train', data_type='video')
    X_test, y_test = extract_features(model, test_dataloader, args, split='test', data_type='video')

    reg = Ridge()
    results = {'mse': [], 'pcc': []}
    for i, dimension in enumerate(['valence', 'arousal']):
        reg.fit(X_train, y_train[:, i])
        y_pred = reg.predict(X_test)
        y_pred = savgol_filter(y_pred, window_length=100, polyorder=3)
        mse = mean_squared_error(y_test[:, i], y_pred)
        pcc = np.corrcoef(y_test[:, i], y_pred)[0, 1]
        results['mse'].append(mse)
        results['pcc'].append(pcc)
        print(f'{dimension}: mse={mse:.4f}, pcc={pcc:.4f}')
    print(f'Average: mse={np.mean(results["mse"]):.4f}, pcc={np.mean(results["pcc"]):.4f}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bold', choices=['bold', 'mg', 'meld', 'emotic', 'la'])
    parser.add_argument('--ckpt-path', type=str, default='./exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt')
    cargs = parser.parse_args()
    args = EvalArgs(
        dataset=cargs.dataset,
        ckpt_path=cargs.ckpt_path,
    )
    
    # basic setup
    set_random_seed(args.seed, deterministic=args.cuda_deterministic)
    logger = setup_logger(name='eval')
    rprint(args)
    
    if args.dataset == 'emotic':
        args.data_type = 'image'
    else:
        args.data_type = 'video'
    # <--- Toni --->
    if not args.use_cache:
        # load pretrained model
        model = EmotionCLIP(
            video_len=args.video_len,
            backbone_checkpoint=None,
        )
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=args.ckpt_strict)
            model.eval().to(args.device if torch.cuda.is_available() else "cpu")
            logger.info(f'Model loaded from {args.ckpt_path}')
        else:
            raise ValueError('No checkpoint provided')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Create dataset
        image_paths = ["./src/images/brunette-woman-smiling.jpg", "./src/images/angry.png", "./src/images/sad.png", "./src/images/angry1.jpg", "./src/images/male_smilling.jpg"]
        test_dataloader = CustomImageDataset(image_paths)
        dataloader = DataLoader(test_dataloader, batch_size=4, shuffle=False)
        
        # 2. Iterate and extract features
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            mask = torch.ones(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=device)

            with torch.no_grad():
                features = model.encode_image(imgs, image_mask=mask)
                features = torch.nn.functional.normalize(features, dim=-1)
            print(features.shape)

        # 3. Classify with prompts
        # prompts = [
        #     "a female is happy",
        #     "a female is sad",
        #     "a female is angry",
        #     "a female is surprised",
        #     "a male is happy",
        #     "a male is sad",
        #     "a male is angry",
        #     "a male is surprised"
        # ]

        prompts = [
            "a person is happy",
            "a person is sad",
            "a person is angry",
            "a person is surprised",
        ]

        # 3. tokenize prompts
        import clip
        tokenized = clip.tokenize(prompts).to(device)
        text_features = []
        with torch.no_grad():
            text_features = model.encode_text(tokenized)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # 4. Iterate over dataloader
        
        for imgs, paths in dataloader:
            imgs = imgs.to(device)
            # dummy masks (all ones)
            mask = torch.ones(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=device)
            with torch.no_grad():
                img_features = model.encode_image(imgs, image_mask=mask)
                img_features = torch.nn.functional.normalize(img_features, dim=-1)

            # compute cosine similarity
            for i, feat in enumerate(img_features):
                sims = torch.matmul(text_features, feat.unsqueeze(-1)).squeeze(-1)
                probs = F.softmax(sims, dim=0)
                best_idx = torch.argmax(probs).item()
                best_prob = probs[best_idx].item() * 100

                print(f"Best Match")
                print(f"{image_paths[i]}: {prompts[best_idx]} ({best_prob:.2f}%)")
                print("\nAll Predictions:")
                prob_pairs = [(j, prompts[j], probs[j].item() * 100) for j in range(len(prompts))]
                prob_pairs.sort(key=lambda x: x[2], reverse=True)

                for rank, (idx, prompt, prob) in enumerate(prob_pairs, 1):
                    marker = "★" if idx == best_idx else " "
                    print(f"{rank:<4} {prompt:<25} {prob:>8.2f}% {marker}")
                print("=" * 50)

    # </--- Toni --->
if __name__ == '__main__':
    main()