from flask import Flask, request, render_template
import pickle
import requests

app = Flask(__name__)

# 🔑 Load model and vectorizer
model = pickle.load(open("lr_model.pkl", "rb"))
vectorizer = pickle.load(open("lr_vectorizer.pkl", "rb"))

# 🌐 Colab Gradio API URL
GRADIO_API_URL = "https://c7d4a2691fe65103a6.gradio.live"

# 🧠 Predict whether news is real or fake
def predict_news(text):
    transformed = vectorizer.transform([text])
    pred_class = model.predict(transformed)[0]  # 0 or 1
    result = "Fake News" if pred_class == 1 else "Real News"
    confidence = model.predict_proba(transformed)[0][pred_class]
    confidence_score = round(confidence * 100, 2)  # convert to percentage
    return result, confidence_score

# 🦙 Get explanation from LLaMA 3
def get_llama_explanation(text, prediction, confidence):
    payload = {
        "data": [text, prediction, str(confidence)]
    }
    response = requests.post(GRADIO_API_URL, json=payload)
    
    print("Raw response:", response.text)  # ✅ This will print in VS Code’s terminal

    response_json = response.json()
    if "data" in response_json:
        return response_json["data"][0]
    else:
        return f"Error: Unexpected response format → {response_json}"


# 🚪 Route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        news_text = request.form["news"]
        prediction, confidence = predict_news(news_text)
        explanation = get_llama_explanation(news_text, prediction, confidence)

        return render_template("index.html",
                               news=news_text,
                               prediction=prediction,
                               confidence=confidence,
                               explanation=explanation)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
