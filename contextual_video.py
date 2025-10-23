import argparse
import os 
import os.path as osp
from dataclasses import dataclass
import logging
import sys
from typing import Literal, Optional
from collections import defaultdict
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import settings
from src.models.base import EmotionCLIP
from src.datasets.meld import MELD
from src.datasets.bold import BoLD
from src.datasets.movie_graphs import MovieGraphsDataset
from src.datasets.emotic import Emotic
from src.datasets.custom_image_dataset import CustomImageDataset
from src.datasets.liris_accede import LirisAccede
from src.engine.utils import set_random_seed
from src.engine.logger import setup_logger
import json
import librosa

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

def main(image_folder):

    # 1. Create dataset
    frame_results  = {}
    frame_results["content"] = []
    frame_rate = 24.0
    segment_number = 0
    segment_duration = 2.0
    image_folder = "./videos1/frames/" + image_folder
    # image_folder = "./videos_test/frames/" + image_folder
    image_paths = [osp.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    print(image_folder)
    test_dataloader = CustomImageDataset(image_paths)
    dataloader = DataLoader(test_dataloader, batch_size=4, shuffle=False)

    # 2. Iterate and extract features
    for imgs, _, paths in dataloader:
        imgs = imgs.to(device)
        mask = torch.ones(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=device)

        with torch.no_grad():
            features = model.encode_image(imgs, image_mask=mask)
            features = torch.nn.functional.normalize(features, dim=-1)
        # print(features.shape)

    # 3. Classify with prompts
#     contextual_prompts = {
#     "happy": ["slightly smiling", "beaming with joy", "grinning broadly"],
#     "sad": ["slightly frowning", "appearing downcast", "teary-eyed but composed"],
#     "angry": ["mildly annoyed", "furious with rage", "showing intense anger"],
#     "surprised": ["slightly startled", "utterly shocked", "wide-eyed and speechless"],
#     "fearful": ["nervous and anxious", "terrified and trembling", "scared but hiding it"],
#     "disgusted": ["mildly disgusted", "extremely repulsed", "grimacing heavily"],
#     "neutral": ["expressionless", "calm and composed", "no particular reaction"]
# }

    contextual_prompts = {
        "happy": [
            "a person smiling happily",
            "joyful facial expression",
            "cheerful face",
            "person with big smile",
            "radiant smile",
            "beaming with happiness",
            "person laughing with joy",
            "delighted expression",
            "bright cheerful eyes",
            "upturned mouth smiling",
            "face full of joy",
            "person with genuine smile",
            "happy and content",
            "smiling warmly",
            "grinning face",
        ],
        "sad": [
            "a person looking sad",
            "sorrowful expression",
            "crying face",
            "person with sad eyes",
            "downturned mouth frowning",
            "tears streaming down face",
            "melancholic expression",
            "grief-stricken face",
            "person looking dejected",
            "mournful expression",
            "sad and forlorn",
            "upset emotional face",
            "person with drooping features",
            "despondent look",
            "tearful eyes",
        ],
        "angry": [
            "a person looking angry",
            "scowling facial expression",
            "irate face",
            "furious eyes",
            "furious facial expression",
            "angry face",
            "person with furrowed brows",
            "red angry face",
            "person scowling",
            "hostile expression",
            "rage-filled face",
            "clenched jaw angry",
            "intense angry glare",
            "person with veins bulging",
            "wrathful expression",
            "person baring teeth angrily",
            "aggressive angry face",
            "seething with anger",
        ],
        "surprised": [
            "a person looking surprised",
            "shocked expression",
            "surprised face",
            "person with wide eyes",
            "mouth open in shock",
            "amazed astonished face",
            "wide-eyed expression",
            "startled look",
            "bewildered expression",
            "taken aback face",
            "person with eyebrows raised",
            "shocked and stunned",
            "expression of wonder",
            "caught off guard face",
            "mouth agape in surprise",
        ],
        "fearful": [
            "a person looking afraid",
            "fearful expression",
            "scared face",
            "person with fear in eyes",
            "terrified expression",
            "eyes wide with fear",
            "panic-stricken face",
            "horrified look",
            "trembling with fear",
            "anxious worried face",
            "person looking frightened",
            "dread-filled expression",
            "nervous fearful eyes",
            "apprehensive face",
            "person recoiling in fear",
        ],
        "disgusted": [
            "a person looking disgusted",
            "repulsed facial expression",
            "disgusted face",
            "person with wrinkled nose",
            "sneering expression",
            "grimacing face",
            "person looking revolted",
            "nauseated expression",
            "face showing aversion",
            "person with curled lip",
            "displeased disgusted look",
            "person turning away in disgust",
            "face contorted in disgust",
            "person showing strong dislike",
        ],
        "neutral": [
            "a person with neutral expression",
            "expressionless face",
            "blank face",
            "person with no emotion",
            "straight-faced expression",
            "calm composed face",
            "flat affect",
            "unreadable expression",
            "stoic neutral face",
            "poker face",
            "person with relaxed features",
            "emotionless blank stare",
            "detached expression",
            "impassive face",
            "person showing no emotion",
        ],
    }


    prompt_to_basic = {}
    for emotion, prompts in contextual_prompts.items():
        for prompt in prompts:
            prompt_to_basic[prompt] = emotion
    # Flatten dictionary into list
    prompts = [prompt for prompts_list in contextual_prompts.values() for prompt in prompts_list]
    # 3. tokenize prompts
    import clip
    tokenized = clip.tokenize(prompts).to(device)
    text_features = []
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    segment = []
    overall_emotions = []
    # 4. Iterate over dataloader
    global_frame_idx = 0
    for imgs, _, paths in dataloader:
        imgs = imgs.to(device)
        # dummy masks (all ones)
        mask = torch.ones(imgs.shape[0], imgs.shape[2], imgs.shape[3], device=device)
        with torch.no_grad():
            img_features = model.encode_image(imgs, image_mask=mask)
            img_features = torch.nn.functional.normalize(img_features, dim=-1)

        # compute cosine similarity
        for i, feat in enumerate(img_features):
            sims = torch.matmul(text_features, feat.unsqueeze(-1)).squeeze(-1)
            temperature = 100.0
            probs = F.softmax(sims * temperature, dim=0)
            prob_pairs = [(j, prompts[j], probs[j].item()) for j in range(len(prompts))]
            segment.append({
                "frame_index": global_frame_idx,
                "predictions": [
                    {
                        "prompt": prompt,
                        "probability": prob,
                    }
                    for rank, (idx, prompt, prob) in enumerate(prob_pairs)
            ]})
            
            if global_frame_idx % (frame_rate * 2) == 0:
                
                emotion_scores = defaultdict(float)
                for frame in segment:
                    for pred in frame['predictions']:
                        emotion_scores[pred['prompt']] += pred['probability']

                # get the best emotion
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                best_prob = emotion_scores[best_emotion] / len(segment)  # average probability across frames
                best_emotion_basic = prompt_to_basic.get(best_emotion, best_emotion)
                # print(f"Best emotion in this 2s segment: {best_emotion_basic} ({best_prob:.3f})")
                print(f"Emotions in this 2s segment:")
                # get top three emotions
                top_three_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                # top_three_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (emotion, score) in enumerate(top_three_emotions):
                    emotion_basic = prompt_to_basic.get(emotion, emotion)
                    print(f"  Top {i+1} emotion: {emotion_basic} {emotion} ({score / len(segment):.3f})")

                # append to results
                frame_results["content"].append({
                    "time": global_frame_idx / frame_rate,
                    "emotion": best_emotion_basic,
                    "confidence": best_prob
                })
                if best_emotion_basic not in overall_emotions:
                    overall_emotions.append(best_emotion_basic)

                # clear the segment buffer for next 2-second window
                segment = []
            global_frame_idx += 1
    return frame_results, overall_emotions
        # Save results to JSON
        # with open("video_result_test.json", "w") as f:
        #     json.dump(frame_results, f, indent=4)

    # </--- Toni --->
if __name__ == '__main__':
    video_folders = sorted([f for f in os.listdir("videos1/frames") if f.lower()])
    print(f"Found {len(video_folders)} video frame folders.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='emotic', choices=['bold', 'mg', 'meld', 'emotic', 'la'])
    parser.add_argument('--ckpt-path', type=str, default='./emotionclip_latest.pt')
    cargs = parser.parse_args()
    args = EvalArgs(
        dataset=cargs.dataset,
        ckpt_path=cargs.ckpt_path,
    )
    
    # basic setup
    set_random_seed(args.seed, deterministic=args.cuda_deterministic)
    logger = setup_logger(name='eval')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    video_results  = {}
    for video_folder in video_folders:
        video_path = f"{video_folder.removesuffix('_frames')}.mp4"
        video_length = round(float(librosa.get_duration(path="videos1/videos/" + video_path)), 2)
        print(f"Processing video folder: {video_folder}")
        frame_results, overall_emotions = main(video_folder)
        video_results[video_path.removesuffix('.mp4')] = {
            "file": video_path,
            "emotions": overall_emotions,
            "content": frame_results["content"],
            "length_seconds": video_length
        }

    with open("video_result_test.json", "w") as f:
        json.dump(video_results, f, indent=4)
