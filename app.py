import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify

# --- Partie 1 & 2 : Chargement et Entraînement ---
df = pd.read_csv("sms_dataset.csv")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]
model = LogisticRegression()
model.fit(X, y)

# --- Partie 3 : Création d’une API ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["message"]
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return jsonify({"classe": prediction[0]})

if __name__ == "__main__":
    # Note: On utilise le port 5000 comme demandé
    app.run(host="0.0.0.0", port=5000)