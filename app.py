import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import spacy

app = Flask(__name__)
CORS(app)

classifier = None
nlp = None
MODEL_PATH = "varshitha-sa/fake-news-distilbert"

def load_models():
    global classifier, nlp
    if classifier is None:
        print("Loading DistilBERT (lazy)...") # <--- ADDED THIS
        classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
        print("DistilBERT loaded successfully.") # <--- ADDED THIS
    if nlp is None:
        print("Loading spaCy (lazy)...") # <--- ADDED THIS
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        print("spaCy loaded successfully.") # <--- ADDED THIS

@app.route('/')
def home():
    return "Fake News Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Load models ONLY when a prediction is requested
    load_models()
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. ML Prediction
    try:
        results = classifier(text[:512])[0]
        best_result = max(results, key=lambda x: x['score'])
        
        raw_label = best_result['label']
        if raw_label == 'LABEL_0':
            prediction = 'FAKE'
        elif raw_label == 'LABEL_1':
            prediction = 'REAL'
        else:
            prediction = raw_label
            
        confidence = best_result['score']
    except Exception as e:
        prediction = "UNKNOWN"
        confidence = 0.0

    # 2. NER (Entities)
    doc = nlp(text[:1000])
    entities = [{"label": ent.label_, "value": ent.text} for ent in doc.ents][:6]

    # 3. Mock SHAP 
    shap_words = []
    words = text.split()[:5]
    for i, w in enumerate(words):
        if len(w) > 3:
            shap_words.append({
                "word": w.replace(".", "").replace(",", ""),
                "value": 0.45 - (i * 0.05),
                "direction": prediction
            })

    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence),
        "entities": entities,
        "shap_words": shap_words
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)