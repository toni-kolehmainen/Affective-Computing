# Modifications for dat collection and preprocessing pipeline for "Learning Emotion Representations from Verbal and Nonverbal Communication".


1. Setup all requirements, download yolov7.pt add it to yolo_additions.
2. Example structure that you need for meld_eval.py, atleast how I understand it.

- test_human_boxes.json
- train_human_boxes.json
- dev_human_boxes.json
- MELD.Raw
    |- dev_sent_emo.csv
    |- test_sent_emo.csv
    |- train_sent_emo.csv
- frames
    |- test_splits
        |- dia0_utt0.mp4
            |- frames
                |- 00000001.jpg
                ...
        |- ...
    |- dev_splits
    |- train_splits

3. csv files from MELD dataset
4. Frames using edited download_videos
- Before anything add CURRENT_PATH\preprocessing and run structure_problem.py. This creates correct stucture for download_videos.
- Only command "python download_videos.py --source 'local' --fps 8" works, so you need to move current data to folder called youtube.
- Repeat for test, train and dev.

5. Bounding boxes
- It creates output for test, train and dev. Copy to root X_human_boxes.json

## Setup

### To setup human detection part, please run the following code and follow their setup.

```
git clone https://github.com/WongKinYiu/yolov7.git
cp yolo_additions/datasets.py yolov7/utils/datasets.py
cp yolo_additions/detect_human.py yolov7/detect_human.py
```

### Then, you may need [ffmpeg](https://ffmpeg.org/) for video processing.

## Run

We use the following default setting but you may change them as needed.

2. Run `download_videos.py` to preprocess videos saved locally:

   ```
   python download_videos.py --source local --save_dir PATH-TO-VIDEO-FOLDER
   ```

   This script saves the frames of the videos at a rate set by the --fps argument. Currently, it will take all the videos in the folder, extract frames, and save them in a subfolder called "frames".

   NOTE: It seems one of the scripts below expects a single folder for each clip in the dataset inside the "frames" folder. Run the rest of the commands to check. This is something that should be changed to get everything working.

   (I think this is not necessary for now) You can optionally store all the frames in hdf5 format using:

   ```
   python frames_to_hdf5.py
   ```

3. Generate human bounding boxes using:

   ```
   cd yolov7
   python detect_human.py --source PATH-TO-VIDEO-FOLDER
   ```

4. Run evaluation. This step is currently failing because folders are not structured as expected:

   ```
   python linear_eval.py --dataset meld --ckpt-path emotionclip_latest.pt
   ```

5. Structure

   Train, test and dev needs frame folder. Only this command work, so videos need to be in youtube folder as following
   - youtube
        |- dia0_utt0.mp4
           |- dia0_utt0.mp4
           |- frame
        |- dia1_utt0.mp4
           |- dia0_utt0.mp4
           |- frame
   ```
   python download_videos.py --source 'local' --fps 8
   ```
   Also, need to have train_human_boxes.json, test_human_boxes.json and dev_human_boxes.json.
   detect_human.py edit to correct structure.
   
