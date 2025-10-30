# Emotion detector

This project is based on EmotionCLIP. This emotion detector uses a custom dataset and the final result is a hue bar where the final user can see the emotions shown during the duration of the clip.
```
For further information about EmotionCLIP and its usage please [click here](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_Emotion_Representations_From_Verbal_and_Nonverbal_Communication_CVPR_2023_paper.pdf).
```

## Overview

![EmotionCLIP](.github/emotionclip_acomp.jpg)

# Setup

1. Create venv
2. Install requirements by running `pip install -r requirements.txt`
3. Download model .pt file

Run command is in commands. Check path to model is correct!!!



## Usage
### Preprocessing
1. Lorem Ipsum
2. Lorem Ipsum
3. Lorem Ipsum


### Video Player
´Prerequisite: pyqt5 ´

1. The application requires two files in the project structure:
´videos/audio_result.json´
´videos/video_result.json´

The video and audio files (recommended format .p4 or .wav) should follow this directory structure:
´videos/video´
´videos/audio´

2. Run the player.py
3. On the interface, type an emotion on a search bar, then hit "Load" to filter the list and load the first matching video. A list of related videos will show on the side.
4. Use the < and > buttons to cycle through the videos
5. Use the "play/pause" button to control the video

# Acknowledgments
Our code is based on [EmotionCLIP](https://github.com/Xeaver/EmotionCLIP).
