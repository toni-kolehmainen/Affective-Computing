"""
Emotion recognition from audio using a pre-trained model. The model should be downloaded when you first run the code. Emotion classes: neutral, calm(*), happy, sad, angry, fearful, disgusted, surprised.

(* calm could be equated with neutral in our context?)

Input: Audio file, define in audio_path variable.
Output: Predicted emotion label with confidence score.

Requires an audio file in WAV format. One way to extract audio from video is using ffmpeg by running the following command in terminal:
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav
"""

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

AUDIO_PATH = "data/audio_example.wav"

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    confidence = probabilities[0, predicted_id].item()
    
    return predicted_label, confidence

print("Using CUDA:", torch.cuda.is_available())
predicted_emotion, confidence = predict_emotion(AUDIO_PATH, model, feature_extractor, id2label)
print(f"Predicted emotion: {predicted_emotion} ({confidence*100:.2f}%)")
