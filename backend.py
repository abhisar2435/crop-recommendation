from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize app
app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return "Crop Recommendation API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract input values
        N = data['N']
        P = data['P']
        K = data['K']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Convert to array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "recommended_crop": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# Run server
if __name__ == '__main__':
    app.run(debug=True)