# plant detection 
import os
import cv2
import numpy as np

# Directory where the images are stored
data_dir = "path_to_data_directory"
categories = ["Healthy", "Diseased"]

# Preprocess and load images
data = []
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)  # Label encoding
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_array = cv2.resize(img_array, (128, 128))  # Resize image
            data.append([img_array, class_num])
        except Exception as e:
            pass

# Shuffle data to ensure randomness
import random
random.shuffle(data)

# Split data into features (X) and labels (y)
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)

# Convert to numpy arrays
X = np.array(X).reshape(-1, 128, 128, 3)
y = np.array(y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Normalize the data
X = X / 255.0

# Convert labels to categorical format
y = to_categorical(y, num_classes=len(categories))

# Define the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np

app = Flask(_name_)

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = np.array(img_array).reshape(-1, 128, 128, 3)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            image_path = "uploads/" + file.filename
            file.save(image_path)
            image = preprocess_image(image_path)
            prediction = model.predict(image)
            predicted_class = categories[np.argmax(prediction)]
            return f'The plant leaf is: {predicted_class}'
    return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)
