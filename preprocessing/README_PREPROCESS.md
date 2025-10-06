# Modifications for dat collection and preprocessing pipeline for "Learning Emotion Representations from Verbal and Nonverbal Communication".

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
   python videos_to_frames.py --input-dir PATH-TO-VIDEO-FOLDER --output-dir PATH-TO-FOLDER

   python videos_to_frames.py --input-dir /Data/Documents/masters/uni_oulu/semester1/affective_computing/project/MELD.Raw/test/mini --output-dir /Data/Documents/masters/uni_oulu/semester1/affective_computing/project/MELD.Raw/test/frames

   ```
   This script saves the frames of the videos at a rate set by the --fps argument. Currently, it will take all the videos in the folder, extract frames, and save them in a folder called "frames".

   For now for MELD it should be something like

   ```

   python videos_to_frames.py --input-dir PATH-TO-VIDEO-FOLDER (for example train/test video folder) --output-dir PATH-TO-FOLDER/MELD.Raw/train/frames

   ```

   Do this step for both train and test.


   (I think this is not necessary for now) You can optionally store all the frames in hdf5 format using:

   ```
   python frames_to_hdf5.py
   ```

3. Generate human bounding boxes using:

   ```
   cd yolov7
   python detect_human.py --source PATH-TO-VIDEO-FOLDER

   ```

   For MELD it should be something like 

   ```
   cd yolov7
   python detect_human.py --source PATH-TO-VIDEO-FOLDER/MELD.Raw/train/frames

   ```

   Do this step for both train and test. Then, change the name of each file to train_human_boxes.json and test_human_boxes.json and paste these files inside MELD.Raw folder (manual step for now).

4. Run evaluation:

   ```
   python meld_linear_eval.py --dataset meld --ckpt-path emotionclip_latest.pt
   ```

   If the folder structure above was followed, it should run smoothly and show metrics at the end.

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
   
