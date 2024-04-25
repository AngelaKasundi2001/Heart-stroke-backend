from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib

app = Flask(__name__)
CORS(app)


# Load the trained model
def load_trained_model():
    model = joblib.load('stroke_detection_model.pkl')
    return model


# Preprocess the image
def preprocess_image(image):
    # Resize image to match model input shape
    image = cv2.resize(image, (224, 224))
    # Preprocess input for VGG16 model
    image = preprocess_input(image)
    return image


## Making Predictions

## The application exposes an endpoint for making predictions:

## - **Endpoint**: `/predict`
## - **Method**: POST
## - **Request Body**: Image file
## - **Response**: JSON with prediction result ("Stroke" or "No Stroke")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    # Get the image file from the request
    file = request.files['image']

    # Read the image file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Load the trained model
    model = load_trained_model()
    # Reshape the preprocessed image for prediction
    # Reshape from (224, 224, 3) to (1, 224*224*3)
    preprocessed_image_flat = preprocessed_image.reshape((1, -1))
    # Make prediction
    prediction = model.predict(preprocessed_image_flat)
    # Map numerical prediction to corresponding label
    result = "Stroke" if prediction == 1 else "No Stroke"
    # Return the prediction
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
