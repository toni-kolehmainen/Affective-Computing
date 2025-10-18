import os
import subprocess
import argparse
from tqdm import tqdm
import re
import json


OUTPUT_JSON = "video_frames_timestamps.json"


def find_videos_recursive(input_dir: str, extensions=(".mp4",)):
    """
    Recursively find all video files with specified extensions in input_dir.
    Returns a list of (full_path, relative_path) tuples.
    """
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_dir)
                video_files.append((full_path, relative_path))
    return video_files


def videos_to_frames(input_dir: str, output_dir: str, fps: int = 24):
    """
    Extract frames from all .mp4 videos in input_dir and subfolders.
    Preserve the folder structure in output_dir and save a single JSON
    mapping video names to frame timestamps.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_paths = find_videos_recursive(input_dir)

    video_name_frames_dict = {}

    for full_video_path, rel_video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
        video_name = os.path.splitext(os.path.basename(rel_video_path))[0]

        rel_folder = os.path.dirname(rel_video_path)

        frames_folder = os.path.join(output_dir, rel_folder, f"{video_name}_frames")
        os.makedirs(frames_folder, exist_ok=True)

        command = [
            "ffmpeg",
            "-i", full_video_path,
            "-vf", f"fps={fps},showinfo",
            "-q:v", "5",
            os.path.join(frames_folder, "%08d.jpg")
        ]

        process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        timestamp_pattern = re.compile(r'n:\s*(\d+)\s+.*pts_time:(\d+\.\d+)')
        frame_name_timestamp_dict = {}

        for line in process.stderr:
            match = timestamp_pattern.search(line)
            if match:
                frame_num = int(match.group(1)) + 1
                timestamp = float(match.group(2))
                frame_filename = f"{frame_num:08d}.jpg"
                frame_name_timestamp_dict[frame_filename] = timestamp

        process.wait()

        video_key = rel_video_path.split("/")[-1].replace(".mp4", "")
        video_name_frames_dict[video_key] = frame_name_timestamp_dict

    output_json_path = os.path.join(output_dir, OUTPUT_JSON)
    with open(output_json_path, "w") as jsonfile:
        json.dump(video_name_frames_dict, jsonfile, indent=2)

    print(f"Saved all frame timestamps to: {output_json_path}")


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos (recursive) and save timestamps")
    parser.add_argument("--input-dir", type=str, required=True, help="Main folder containing videos (recursively)")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder to save extracted frames and JSON")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second to extract")
    args = parser.parse_args()
    videos_to_frames(args.input_dir, args.output_dir, args.fps)
