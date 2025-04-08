from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for handling cross-origin requests
import joblib
import numpy as np

# Create Flask app instance
app = Flask(__name__)

# Enable CORS to allow frontend at port 5500 to communicate with the backend
CORS(app, origins=["http://127.0.0.1:5500"])  # Replace with your frontend URL if necessary

# Load the pre-trained model (replace with your actual model path)
model = joblib.load("diabetes_model.pkl")  # Make sure your model file is in the same directory or adjust the path

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent by the frontend
        data = request.get_json()

        features = np.array([[
    int(data["HighBP"]),
    int(data["HighChol"]),
    int(data["CholCheck"]),
    float(data["BMI"]),
    int(data["Smoker"]),
    int(data["Stroke"]),
    int(data["HeartDiseaseorAttack"]),
    int(data["PhysActivity"]),
    int(data["Fruits"]),
    int(data["Veggies"]),
    int(data["HvyAlcoholConsump"]),
    int(data["AnyHealthcare"]),
    int(data["NoDocbcCost"]),
    int(data["GenHlth"]),
    float(data["MentHlth"]),
    float(data["PhysHlth"]),
    int(data["DiffWalk"]),
    int(data["Sex"]),
    int(data["Age"]),
    int(data["Education"]),
    int(data["Income"])
]])


        # Make a prediction using the trained model
        prediction = model.predict(features)

        # Return the prediction result
        # You can return the prediction as an integer (0, 1, 2 for no diabetes, prediabetes, diabetes)
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        # Handle potential errors and return a message
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
