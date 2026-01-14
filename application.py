from flask import Flask, request,jsonify
from src.customerchurn.pipelines.prediction_pipeline import PredictionPipeline


application = Flask(__name__)

predictor = PredictionPipeline()

@application.get('/')
def home():
    return("<h1>Welcome to Churn prediction App</h1>")

@application.get('/health')
def health():
    return jsonify({'status':"ok"})

@application.post('/predict')
def predict():
    payload = request.get_json()
    if payload is None:
        return jsonify({"error":"Invalid JSON"}), 400
    
    result = predictor.predict(payload)
    return jsonify(result)

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=True)
    