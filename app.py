
import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="GoEmotions Emotion Detector", layout="centered")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json
import numpy as np



import os

def reassemble_model(chunk_prefix="model_chunk_", folder="goemotions_model", output_path="model.safetensors"):
    # Ensure the folder exists
    if not os.path.exists(folder):
        print(f"⚠️ The folder {folder} does not exist.")
        return

    # Ensure the output file is within the same folder
    output_path = os.path.join(folder, output_path)
    
    # Check if the output file already exists
    if not os.path.exists(output_path):
        print("🔧 Reassembling model from chunks...")
        with open(output_path, 'wb') as out_file:
            i = 0
            while True:
                # Construct the chunk filename (e.g., model_chunk_aa, model_chunk_ab, ...)
                chunk_file = os.path.join(folder, f"{chunk_prefix}{chr(97 + i // 26)}{chr(97 + i % 26)}")
                if not os.path.exists(chunk_file):
                    print(f"⚠️ Missing chunk: {chunk_file}")
                    break
                else:
                    print(f"Reading chunk: {chunk_file}")
                    with open(chunk_file, 'rb') as f:
                        chunk_data = f.read()
                        if len(chunk_data) > 0:
                            out_file.write(chunk_data)
                        else:
                            print(f"⚠️ Empty chunk detected: {chunk_file}")
                i += 1
        
        # Check if the file size matches the expected size
        reassembled_size = os.path.getsize(output_path)
        print(f"Reassembled model size: {reassembled_size} bytes")
        
        if reassembled_size > 0:
            print("✅ Model reassembled successfully.")
        else:
            print("⚠️ Model reassembly failed, size is 0 bytes!")
    else:
        print(f"⚠️ {output_path} already exists. Skipping reassembly.")

# Usage: Reassemble the model chunks into the same folder (goemotions_model)
reassemble_model(chunk_prefix="model_chunk_", folder="goemotions_model", output_path="model.safetensors")


# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("goemotions_model")
    tokenizer = AutoTokenizer.from_pretrained("goemotions_model")
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# Emotion mapping based on your provided columns in dataset (go emotion)
emotion_mapping = {
    "LABEL_0": "admiration",
    "LABEL_1": "amusement",
    "LABEL_2": "anger",
    "LABEL_3": "annoyance",
    "LABEL_4": "approval",
    "LABEL_5": "caring",
    "LABEL_6": "confusion",
    "LABEL_7": "curiosity",
    "LABEL_8": "desire",
    "LABEL_9": "disappointment",
    "LABEL_10": "disapproval",
    "LABEL_11": "disgust",
    "LABEL_12": "embarrassment",
    "LABEL_13": "excitement",
    "LABEL_14": "fear",
    "LABEL_15": "gratitude",
    "LABEL_16": "grief",
    "LABEL_17": "joy",
    "LABEL_18": "love",
    "LABEL_19": "nervousness",
    "LABEL_20": "optimism",
    "LABEL_21": "pride",
    "LABEL_22": "realization",
    "LABEL_23": "relief",
    "LABEL_24": "remorse",
    "LABEL_25": "sadness",
    "LABEL_26": "surprise",
    "LABEL_27": "neutral"
}

# Load the config.json from the model
with open("goemotions_model/config.json", "r") as f:
    config = json.load(f)
    # Update the id2label mapping to use the new emotion labels
    id2label = {int(k): emotion_mapping[v] for k, v in config["id2label"].items()}

# UI
st.title("😃 GoEmotions Emotion Classifier")
st.markdown("Enter a sentence to detect **multiple emotions** using your trained model.")

text = st.text_area("Enter text here:", height=150)

threshold = st.slider("Prediction threshold", 0.2, 0.9, 0.3, 0.01)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)[0]

        pred_emotions = {
            id2label[i]: float(prob)
            for i, prob in enumerate(probs)
            if prob >= threshold
        }

        if pred_emotions:
            st.subheader("Predicted Emotions:")
            for emotion, score in sorted(pred_emotions.items(), key=lambda x: -x[1]):
                st.write(f"✅ {emotion} — `{score:.2f}`")
        else:
            st.info("No strong emotions detected above threshold.")
