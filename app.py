from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)

# Load the Gradient Boosting model using joblib
model = joblib.load('the_model.pkl')

# Initialize the MediaPipe FaceMesh model


# Function to extract facial landmarks from an image
def extract_facial_landmarks(image):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        features = np.array([landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks])
        return features
    else:
        return None

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from POST request
    image_file = request.files['image']

    # Read image file using OpenCV
    image_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is not None:
        # Extract facial landmarks from the image
        landmarks = extract_facial_landmarks(image)
        # resized_image = np.array(image.resize((224, 224)))

        if landmarks is not None:
            # Perform prediction
            prediction = model.predict(landmarks.reshape(1, -1))
            # Convert prediction to a human-readable label
            label = "Stroke" if prediction == 1 else "Non-Stroke"
            return jsonify({'prediction': label})
        else:
            return jsonify({'error': 'Facial landmarks not detected in the image'})
    else:
        return jsonify({'error': 'Failed to read the image file'})


if __name__ == '__main__':
    app.run()
