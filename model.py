import cv2
import numpy as np
from keras.models import load_model

class EmotionRecognitionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, face_image):
        # Preprocess the face image
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image / 255.0  # Normalize
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = np.expand_dims(face_image, axis=0)
        # Make predictions
        predictions = self.model.predict(face_image)
        return predictions
