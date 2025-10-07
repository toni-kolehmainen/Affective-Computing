import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QToolTip, QLineEdit
from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtGui import QPainter, QColor, QFont

class EmotionTimeline(QWidget):
    def __init__(self, video_duration=100):
        super().__init__()
        self.setMouseTracking(True)
        self.video_duration = video_duration  # total seconds
        # Example emotion segments [(start, end, color, label)]
        self.emotions = [
            (0, 20, QColor("#FF6666"), "Angry"),
            (20, 50, QColor("#66CCFF"), "Sad"),
            (50, 80, QColor("#66FF66"), "Joyful"),
            (80, 100, QColor("#FFD966"), "Hopeful")
        ]
        self.setMinimumHeight(150)

    def paintEvent(self, event):
        painter = QPainter(self)
        bar_height = self.height() // 2
        y = (self.height() - bar_height) // 2
        total_width = self.width()

        # Draw each segment
        for (start, end, color, label) in self.emotions:
            start_x = int((start / self.video_duration) * total_width)
            end_x = int((end / self.video_duration) * total_width)
            rect = QRectF(start_x, y, end_x - start_x, bar_height)
            painter.fillRect(rect, color)

    def mouseMoveEvent(self, event):
        # Map mouse x position to video time
        pos = event.pos()
        total_width = self.width()
        time = int((pos.x() / total_width) * self.video_duration)

        # Find emotion at this position
        emotion_label = ""
        for (start, end, color, label) in self.emotions:
            if start <= time <= end:
                emotion_label = label
                break

        # Show tooltip
        if emotion_label:
            QToolTip.setFont(QFont("SansSerif", 10))
            QToolTip.showText(event.globalPos(), f"{time}s - {emotion_label}", self)


class UserInput(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setPlaceholderText("Type your search here...")

        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self, text):
        print(f"User typed: {text}") # Placeholder for actual search logic
        
def main():
    app = QApplication([])
    widget = QWidget()
    widget.resize(600, 100)
    widget.setWindowTitle("Emotion Timeline Demo")
    layout = QVBoxLayout()

    # Add user input field
    user_input = UserInput()
    layout.addWidget(user_input)
    # Add timeline
    timeline = EmotionTimeline()
    layout.addWidget(timeline)

    widget.setLayout(layout)
    widget.show()
    app.exec_()

if __name__ == "__main__":
    main()
