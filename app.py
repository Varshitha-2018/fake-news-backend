import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import spacy

app = Flask(__name__)
CORS(app)

# Do NOT load models here. Just set up variables.
classifier = None
nlp = None
MODEL_PATH = "varshitha-sa/fake-news-distilbert"

def load_models():
    global classifier, nlp
    if classifier is None:
        print("Loading DistilBERT (lazy)...")
        classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
        print("Model loaded.")
    if nlp is None:
        print("Loading spaCy (lazy)...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        print("spaCy loaded.")

@app.route('/')
def home():
    return "Fake News Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Load models ONLY when a prediction is requested
    load_models()
    
    data = request.json
    text = data.get('text', '')
    # ... rest of your prediction logic stays the same ...