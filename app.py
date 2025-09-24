from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import json
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperModel
import os
from pydub import AudioSegment
import io
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")


# ======================
# Classifier Head
# ======================
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.net(x)

# ======================
# Load config and weights
# ======================
MODEL_ID = "openai/whisper-large-v3-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

with open("models/whisper_classifier_config.json", "r") as f:
    config = json.load(f)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
whisper_model = WhisperModel.from_pretrained(MODEL_ID).to(DEVICE)
whisper_model.eval()

model = ClassifierHead(
    input_dim=config["input_dim"],
    num_labels=config["num_labels"],
    hidden_dim=config["hidden_dim"],
    dropout=config["dropout"]
).to(DEVICE)

model.load_state_dict(torch.load("models/whisper_classifier_head.pth", map_location=DEVICE))
model.eval()

# ======================
# Inference functions
# ======================
def extract_embedding(file_path, sr=SR):
    waveform, _ = librosa.load(file_path, sr=sr)
    inputs = feature_extractor([waveform], sampling_rate=sr, return_tensors="pt").to(DEVICE)

    feats = inputs.input_features
    B, n_mels, T = feats.shape

    if T < 3000:
        pad_size = 3000 - T
        feats = F.pad(feats, (0, pad_size), mode="constant", value=0.0)
    elif T > 3000:
        feats = feats[:, :, :3000]

    with torch.no_grad():
        encoder_outputs = whisper_model.encoder(feats)
        hidden_states = encoder_outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1)

    return embedding

def predict(file_path):
    embedding = extract_embedding(file_path)
    with torch.no_grad():
        logits = model(embedding)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    result = {label: float(prob) for label, prob in zip(config["labels"], probs)}
    pred_label = max(result, key=result.get)
    return pred_label, result

# ======================
# Flask App
# ======================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    # Load the audio
    try:
        audio = AudioSegment.from_file(file)
    except Exception as e:
        return jsonify({"error": f"Could not read audio: {e}"}), 400

    # Change to wav
    temp_wav = "uploads/tmp_audio.wav"
    audio.export(temp_wav, format="wav")

    # Make the prediction
    try:
        pred_label, result = predict(temp_wav)
        return jsonify({"prediction": pred_label, "probabilities": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)