from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import os
import argparse
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
VIDEO_FOLDER = "videos"
AUDIO_FOLDER = "audio"
CHUNK_DURATION = 2.0  # seconds

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

from emotion_from_audio_chunks import predict_emotions_for_chunks

def main(args: argparse.Namespace):
    # --- Extract audio from videos in VIDEO_FOLDER ---
    if args.audio:
        os.makedirs(AUDIO_FOLDER, exist_ok=True)
        video_files = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp4")])
        print(f"Found {len(video_files)} video files in '{VIDEO_FOLDER}' folder.")
        for video_file in video_files:
            clip = VideoFileClip(os.path.join(VIDEO_FOLDER, video_file))
            if clip.audio is None:
                print(f"⚠️ No audio found in video '{video_file}', skipping.")
                continue
            audio = clip.audio
            audio.write_audiofile(os.path.join(AUDIO_FOLDER, f"{video_file}.wav"))

    # --- Predict emotions for audio files in AUDIO_FOLDER ---
    audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".wav")])
    print(f"Found {len(audio_files)} audio files in '{AUDIO_FOLDER}' folder.")

    audio_results = {}
    for itr, audio_file in enumerate(audio_files):
        # if itr == 4:
        #     break
        audio_path = os.path.join(AUDIO_FOLDER, audio_file)
        emotions = predict_emotions_for_chunks(audio_path, model, feature_extractor, id2label, chunk_duration=CHUNK_DURATION)
        print(f"Predictions for '{audio_file}':")

        overall_emotions = []
        # Build the content list
        content_list = []
        for start_time, emotion, confidence in emotions:
            content_list.append({
                "time": float(start_time),
                "emotion": emotion,
                "confidence": float(confidence)
            })
            if emotion not in overall_emotions:
                overall_emotions.append(emotion)

        # Add to main JSON object
        audio_results[audio_file] = {
            "file": audio_file,
            "emotions": overall_emotions,
            "content": content_list,
            "length_seconds": float(librosa.get_duration(path=audio_path))
        }
    
    # Write the results to audio_result.json
    with open("audio_result.json", "w") as f:
        json.dump(audio_results, f, indent=4)

    print("✅ Results written to audio_result.json")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract audio from videos and predict emotions")
    # p.add_argument('--file', type=str, help="File to process")
    p.add_argument('--audio', action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Using CUDA:", torch.cuda.is_available())
    main(args)

# example usage:
# python audio_json_writer.py --audio
# python audio_json_writer.py --no-audio