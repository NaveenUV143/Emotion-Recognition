# app.py
import cv2
import numpy as np
from emotions import emotions
from model import EmotionRecognitionModel

# Load pre-trained model
model = EmotionRecognitionModel('model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected.")
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray_frame[y:y + h, x:x + w]

        # Check the shape of the face image
        print(f"Face shape: {face.shape}")  # Debugging: Check face shape

        # Check if the face is of the expected size
        if face.shape[0] != 0 and face.shape[1] != 0:  # Ensure face is not empty
            try:
                # Resize face to the model's expected input size (48x48)
                face_resized = cv2.resize(face, (48, 48))
                # Ensure the resized face is grayscale
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                face_resized = face_resized / 255.0  # Normalize
                face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
                face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

                # Predict emotion
                predictions = model.predict(face_resized)
                if predictions is not None:  # Ensure predictions are valid
                    emotion_index = np.argmax(predictions[0])
                    emotion = emotions[emotion_index]

                    # Draw rectangle around face and display emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                print(f"Error processing face: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
