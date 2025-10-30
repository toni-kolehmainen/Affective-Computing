"""
Emotion recognition from audio using a pre-trained model. The model should be downloaded when you first run the code. Emotion classes: neutral, calm(*), happy, sad, angry, fearful, disgusted, surprised.

(* calm could be equated with neutral in our context?)

Inputs: Audio file and chunk duration.
Outputs: Start time and predicted emotion label with confidence score for each chunk.

Requires an audio file in WAV format. One way to extract audio from video is using ffmpeg by running the following command in terminal:
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav
"""

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np


AUDIO_PATH = "data/audio_example.wav"
CHUNK_DURATION = 2.0  # seconds

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

def preprocess_audio_chunk(audio_array, feature_extractor, max_length):
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
    # Pad input_features to length 3000 if needed
    input_features = inputs["input_features"]
    target_len = 3000
    current_len = input_features.shape[-1]
    if current_len < target_len:
        pad_width = target_len - current_len
        # Pad along the last dimension
        input_features = torch.nn.functional.pad(input_features, (0, pad_width))
        inputs["input_features"] = input_features
    return inputs

def predict_emotion_chunk(audio_array, model, feature_extractor, id2label, chunk_duration=CHUNK_DURATION):
    max_length = int(feature_extractor.sampling_rate * chunk_duration)
    inputs = preprocess_audio_chunk(audio_array, feature_extractor, max_length)
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

def predict_emotions_for_chunks(audio_path, model, feature_extractor, id2label, chunk_duration=CHUNK_DURATION):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    total_length = len(audio_array)
    chunk_size = int(feature_extractor.sampling_rate * chunk_duration)
    num_chunks = int(np.ceil(total_length / chunk_size))
    results = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_array[start:end]
        emotion, confidence = predict_emotion_chunk(chunk, model, feature_extractor, id2label, chunk_duration)
        results.append((i * chunk_duration, emotion, confidence))
    return results

print("Using CUDA:", torch.cuda.is_available())
emotions = predict_emotions_for_chunks(AUDIO_PATH, model, feature_extractor, id2label, chunk_duration=CHUNK_DURATION)
for start_time, emotion, confidence in emotions:
    print(f"Time {start_time:.1f}s - predicted emotion: {emotion} ({confidence*100:.2f}%)")
