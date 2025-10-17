import os
import subprocess
import argparse
from tqdm import tqdm
import re
import json


OUTPUT_JSON = "video_frames_timestamps.json"


def videos_to_frames(input_dir: str, output_dir: str, fps: int = 24):
    """
    Extract frames from all .mp4 videos in input_dir and save them in output_dir.
    Also saves a single JSON with all frame timestamps grouped by video.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
    
    video_name_frames_dict = {}
    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_dir, f"{video_name}_frames")
        os.makedirs(frames_folder, exist_ok=True)

        command = [
            "ffmpeg",
            "-i", video_path,
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

        video_name_frames_dict[video_name] = frame_name_timestamp_dict

    output_json_path = os.path.join(output_dir, OUTPUT_JSON)
    with open(output_json_path, "w") as jsonfile:
        json.dump(video_name_frames_dict, jsonfile, indent=2)

    print(f"Saved all frame timestamps to: {output_json_path}")


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos using FFmpeg and save timestamps")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing MP4 videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder to save extracted frames and JSON")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second to extract")
    args = parser.parse_args()
    videos_to_frames(args.input_dir, args.output_dir, args.fps)
