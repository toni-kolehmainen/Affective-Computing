from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QMainWindow, QLineEdit
)
from PyQt5.QtGui import QIcon, QFontMetrics
import sys
import argparse
import json
import os

AUDIO_RESULT_PATH = "audio_result.json"
AUDIO_FOLDER_PATH = "audio"
VIDEO_FOLDER_PATH = "videos"

emotion_colors = {
    "happy": "#0DFF00",
    "sad": "#FA652F",
    "angry": "#FF0000",
    "fearful": "#800080",
    "neutral": "#BDBDBD",
    "surprised": "#FFBF00",
    "disgust": "#90E104"
}

class VideoPlayer(QMainWindow):
    def __init__(self, audio_results):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Player with Side Panel")
        self.currentFile = ''
        self.current_file_index = -1
        self.listOfFiles = []
        self.audio_results = audio_results
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # --- Search input ---
        self.searchInput = QLineEdit()
        self.searchInput.setPlaceholderText("Search video...")
        self.searchInput.returnPressed.connect(self.search_video)  # trigger on Enter

        self.searchButton = QPushButton("Load")
        self.searchButton.clicked.connect(self.search_video)

        searchLayout = QHBoxLayout()
        searchLayout.addWidget(self.searchInput)
        searchLayout.addWidget(self.searchButton)

        # --- Video widget ---
        videoWidget = QVideoWidget()

        # --- Play button ---
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        # --- Position slider ---
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # --- Error label ---
        self.error = QLabel()
        self.error.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # --- Control layout (play button + slider) ---
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        # --- Main video layout (video + controls + error) ---
        videoLayout = QVBoxLayout()
        videoLayout.addLayout(searchLayout)
        videoLayout.addWidget(videoWidget)
        videoLayout.addLayout(controlLayout)
        videoLayout.addWidget(self.error)

        # --- Side panel layout (two text labels) ---
        sidePanel = QVBoxLayout()
        sidePanel.setContentsMargins(10, 10, 10, 10)

        self.labelTitle = QLabel(f"Video Title: {self.currentFile}")
        self.labelTitle.setStyleSheet("font-weight: bold; font-size: 14px;")
        # self.labelVideoPrediction = QLabel("Prediction: Happy 0.9565")
        self.labelVideoPrediction = QLabel("Video Prediction: NEEDS TO BE IMPLEMENTED")
        self.labelVideoPrediction.setWordWrap(True)
        self.labelAudioPrediction = QLabel("Audio Prediction:")
        self.labelAudioPrediction.setWordWrap(True)
        buttonLayout = QHBoxLayout()
        self.nextButton = QPushButton(">")
        self.nextButton.clicked.connect(self.go_next_video)
        self.previousButton = QPushButton("<")
        self.previousButton.clicked.connect(self.go_previous_video)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.previousButton)
        buttonLayout.addWidget(self.nextButton)

        sidePanel.addWidget(self.labelTitle)
        sidePanel.addWidget(self.labelVideoPrediction)
        sidePanel.addWidget(self.labelAudioPrediction)
        sidePanel.addLayout(buttonLayout) 
        sidePanel.addStretch(1)

        # --- Combine main and side panel in horizontal layout ---
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(videoLayout, stretch=3)  # main video area
        mainLayout.addLayout(sidePanel, stretch=1)    # side text area

        # --- Central widget ---
        wid = QWidget(self)
        wid.setLayout(mainLayout)
        self.setCentralWidget(wid)

        # --- Connect signals ---
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
    def go_next_video(self):
        if not self.listOfFiles:
            return
        self.current_file_index = (self.current_file_index + 1) % len(self.listOfFiles)
        next_file = self.listOfFiles[self.current_file_index]
        self.load_video(next_file)

    def go_previous_video(self):
        if not self.listOfFiles:
            return
        self.current_file_index = (self.current_file_index - 1) % len(self.listOfFiles)
        prev_file = self.listOfFiles[self.current_file_index]
        self.load_video(prev_file)

    def get_slider_gradient(self, slider, content):
        if not content or slider.maximum() == 0:
            return ""
        max_time = self.video_duration
        stops = []

        for i, chunk in enumerate(content):
            start_percent = chunk["time"] / max_time
            end_time = content[i]["time"] if i < len(content) else max_time
            end_percent = end_time / max_time
            if (self.searchInput.text().lower() != "" and
                self.searchInput.text().lower() not in chunk["emotion"]):
                color = "#CCCCCC"  # Grey out non-matching emotions
            else:
                color = emotion_colors.get(chunk["emotion"], "#FFFFFF")
            
            # Thin black separator
            if i != 0:
                if chunk["emotion"] != content[i - 1]["emotion"]:
                    stops.append(f"stop:{start_percent - 0.001:.3f} ""#CCCCCC")
                else:
                    stops.append(f"stop:{start_percent - 0.001:.3f} {color}")
                stops.append(f"stop:{start_percent:.3f} {color}")
            else:
                stops.append(f"stop:{start_percent:.3f} {color}")
            
            stops.append(f"stop:{end_percent:.3f} {color}")

        gradient_str = ", ".join(stops)
        style = f"""
        QSlider::groove:horizontal {{
            border: 1px solid #999999;
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {gradient_str});
            margin: 0px;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: #444444;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        """
        return style

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        # Current emotion label based on video position
        if hasattr(self, 'current_emotion_chunks') and self.current_emotion_chunks:
            pos_sec = position / 1000
            current_chunk = None
            for chunk in self.current_emotion_chunks:
                if pos_sec >= chunk["time"]:
                    current_chunk = chunk
                else:
                    break
            if current_chunk:
                emotion = current_chunk["emotion"]
                confidence = f"{current_chunk['confidence']:.5f}"
                self.labelAudioPrediction.setText(f"Audio Prediction: {emotion}, {confidence}")

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
        if hasattr(self, 'current_emotion_chunks'):
            style = self.get_slider_gradient(self.positionSlider, self.current_emotion_chunks)
            
            # Set initial handle color to first chunk
            # first_emotion = self.current_emotion_chunks[0]["emotion"]
            first_emotion = "neutral"
            style = style.replace("background: #444444;", f"background: {emotion_colors.get(first_emotion, '#444444')};")
            self.positionSlider.setStyleSheet(style)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.error.setText("Error: " + self.mediaPlayer.errorString())

    def search_video(self):
        self.listOfFiles = []  # Reset the list of files
        query = self.searchInput.text().lower()

        # Find matching emotions
        for file_name in self.audio_results.keys():
            _file_name = file_name.removesuffix(".wav")
            if query in self.audio_results[file_name]["emotions"]:
                self.listOfFiles.append(_file_name)
        if len(self.listOfFiles) != 0:
            self.current_file_index = 0
            self.load_video(self.listOfFiles[0])
        else:
            self.error.setText(f"No video found for '{query}'")

    def load_video(self, file_name):
        print(f"Loading video file: {file_name}")
        video_path = os.path.join(VIDEO_FOLDER_PATH, file_name)
        print(f"Loading video: {video_path}")
        self.current_file = file_name
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.playButton.setEnabled(True)
        self.set_video_title(f"Video Title: {file_name}")

        self.current_emotion_chunks = self.audio_results[file_name + ".wav"]["content"]
        self.video_duration = self.audio_results[file_name + ".wav"]["length_seconds"]
        # Show first chunk's emotion and confidence
        first_chunk = self.current_emotion_chunks[0]
        emotion = first_chunk["emotion"]
        confidence = f"{first_chunk['confidence']:.5f}"
        self.labelAudioPrediction.setText(f"Audio Prediction: {emotion}, {confidence}")
        self.error.setText("")
    
    def set_video_title(self, full_title, max_width=200):
        font_metrics = QFontMetrics(self.labelTitle.font())
        elided_text = font_metrics.elidedText(full_title, Qt.ElideRight, max_width)
        self.labelTitle.setText(elided_text)
        self.labelTitle.setToolTip(full_title)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyQt5 Video Player with Side Panel")
    # p.add_argument('--file', type=str, help="File to process")
    return p.parse_args()

def main():
    args = parse_args()
    with open("audio_result.json", "r") as f:
        audio_results = json.load(f)
    app = QApplication(sys.argv)
    videoplayer = VideoPlayer(audio_results)
    videoplayer.resize(800, 480)
    videoplayer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# example usage:
# python player.py
