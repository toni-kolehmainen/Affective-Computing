import torch
import numpy as np
import os
import argparse
import json
from collections import defaultdict

CHUNK_DURATION = 2.0  # seconds

# Paths (change if needed)
SCORES_JSON_PATH = "videos/emotion_scores.json"
TIMESTAMPS_JSON_PATH = "videos/video_frames_timestamps.json"
OUTPUT_JSON = "videos/video_result.json"


def get_emotions_for_chunks(folder_scores, folder_timestamps, chunk_duration=2.0):
    """
    Average emotion scores across frames in each time chunk for a single video folder.
    Returns: (List of (start_time, top_emotion, avg_confidence), max_time)
    """
    frame_data = []

    for frame_name, frame_info in folder_scores.items():
        timestamp = folder_timestamps.get(frame_name)
        if timestamp is None:
            continue

        frame_data.append({
            "timestamp": timestamp,
            "scores": frame_info["scores"]
        })

    if not frame_data:
        return [], 0.0

    # Sort by time
    frame_data.sort(key=lambda x: x["timestamp"])
    max_time = max(item["timestamp"] for item in frame_data)

    chunk_emotions = []
    current_time = 0.0

    while current_time < max_time:
        next_time = current_time + chunk_duration
        chunk_items = [item for item in frame_data if current_time <= item["timestamp"] < next_time]

        if chunk_items:
            # Aggregate and average scores
            emotion_sums = defaultdict(float)
            for item in chunk_items:
                for emotion, score in item["scores"].items():
                    emotion_sums[emotion] += score
            num_items = len(chunk_items)
            emotion_avgs = {emo: total / num_items for emo, total in emotion_sums.items()}
            top_emotion = max(emotion_avgs.items(), key=lambda x: x[1])[0]
            confidence = emotion_avgs[top_emotion]

            chunk_emotions.append((current_time, top_emotion, confidence))
        else:
            chunk_emotions.append((current_time, "neutral", 0.0))

        current_time = next_time

    return chunk_emotions, max_time


def main(args: argparse.Namespace):
    # Load JSON inputs
    with open(SCORES_JSON_PATH, "r") as f:
        emotion_scores = json.load(f)

    with open(TIMESTAMPS_JSON_PATH, "r") as f:
        frames_timestamps = json.load(f)

    video_results = {}

    # Loop through each folder (i.e., video)
    for folder_name, folder_scores in emotion_scores.items():
        if folder_name not in frames_timestamps:
            print(f"⚠️ Skipping {folder_name} (no timestamps found)")
            continue

        folder_timestamps = frames_timestamps[folder_name]

        chunk_emotions, video_length = get_emotions_for_chunks(
            folder_scores, folder_timestamps, chunk_duration=CHUNK_DURATION
        )

        if not chunk_emotions:
            print(f"⚠️ No valid frames for {folder_name}")
            continue

        # Build content for output
        content_list = []
        overall_emotions = set()

        for start_time, emotion, confidence in chunk_emotions:
            content_list.append({
                "time": float(start_time),
                "emotion": emotion,
                "confidence": float(confidence)
            })
            overall_emotions.add(emotion)

        file_name = folder_name.replace("_frames", "")
        video_results[file_name] = {
            "file": folder_name + ".mp4",
            "emotions": list(overall_emotions),
            "content": content_list,
            "length_seconds": float(video_length)
        }

    # Save output JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(video_results, f, indent=4)

    print(f"✅ Results written to {OUTPUT_JSON}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate frame emotion scores into per-chunk timeline JSON.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Using CUDA:", torch.cuda.is_available())
    main(args)
