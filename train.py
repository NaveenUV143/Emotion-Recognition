# emotion_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('fer2013.csv')

# Prepare the data
X = []
y = []

for index, row in data.iterrows():
    # Each image is a string of pixel values
    pixels = row['pixels'].split(' ')
    X.append(np.array(pixels, dtype='float32'))
    y.append(row['emotion'])

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the pixel values to [0, 1]
X = X / 255.0

# Reshape to match the model input
X = X.reshape(-1, 48, 48, 1)  # 48x48 pixels and 1 channel (grayscale)

# One-hot encode the labels
y = to_categorical(y, num_classes=7)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.h5')

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()
