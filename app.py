import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Load spaCy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- LIGHTWEIGHT ML MODEL ---
# We train a quick TF-IDF + Logistic Regression model on the fly
fake_data = [
    "shocking secret cure big pharma hiding truth conspiracy",
    "5g towers cause covid vaccine microchip bill gates exposed",
    "miracle cure discovered doctors hate this"
]
real_data = [
    "federal reserve raises interest rates inflation",
    "new study shows climate change impacts economy",
    "government announces new infrastructure bill"
]

X_train = fake_data + real_data
y_train = [1, 1, 1, 0, 0, 0] # 1 = FAKE, 0 = REAL

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
# ----------------------------

@app.route('/')
def home():
    return "Fake News Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. ML Prediction
    try:
        text_vec = vectorizer.transform([text])
        prob = model.predict_proba(text_vec)[0]
        prediction = 'FAKE' if prob[1] > 0.5 else 'REAL'
        confidence = float(max(prob))
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
        "confidence": confidence,
        "entities": entities,
        "shap_words": shap_words
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)