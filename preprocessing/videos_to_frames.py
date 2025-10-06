import os
import subprocess
import argparse
from tqdm import tqdm

def videos_to_frames(input_dir: str, output_dir: str, fps: int = 24):
    """
    Extract frames from all .mp4 videos in input_dir and save them in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
    
    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_dir, f"{video_name}_frames")
        os.makedirs(frames_folder, exist_ok=True)

        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "5",
            os.path.join(frames_folder, "%08d.jpg")
        ]

        # Run ffmpeg and suppress its internal output to reduce clutter
        status = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if status != 0:
            print(f"⚠️  FFmpeg failed for {video_file}")


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos using FFmpeg")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing MP4 videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder to save extracted frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second to extract")
    args = parser.parse_args()
    videos_to_frames(args.input_dir, args.output_dir, args.fps)  # you can set fps=8 for 8 frames per second
