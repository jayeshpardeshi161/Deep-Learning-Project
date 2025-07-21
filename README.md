# 🧠 Brain Tumor Detection Using CNN (MRI Classification)

## 👨‍🔬 Author Note
This project represents a practical implementation of a binary classification pipeline for medical imaging. It leverages a custom-trained Convolutional Neural Network (CNN) to detect brain tumors in MRI scans.

## 📌 Project Overview
The goal of this project was to build a binary image classification model that predicts whether an MRI scan shows a tumor or no tumor.

## 🗂️ Project Workflow
### ✅ 1. Import Required Libraries

import os, cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input

### purpose :
These are core libraries for image preprocessing, modeling (TensorFlow/Keras), and dataset handling (NumPy, sklearn).

### ✅ 2. Dataset Loading
python

image_directory = 'datasets/'
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

What I did:
Loaded .jpg images from two folders: yes/ and no/.

Assigned labels: 1 for tumor, 0 for no tumor.

# Resizing all images to a uniform shape
INPUT_SIZE = 64

for image_name in no_tumor_images + yes_tumor_images:
    if image_name.endswith('.jpg'):
        label = 0 if image_name in no_tumor_images else 1
        img_path = os.path.join(image_directory, 'no' if label == 0 else 'yes', image_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = img.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(img))
            labels.append(label)

### purpose: 
Standardizing input size improves training efficiency and model consistency.

## ✅ 3. Data Preprocessing
Python

dataset = np.array(dataset)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

### Normalize pixel values [0-255] ➝ [0-1]
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

### purpose: 
Train/test split allows generalization testing. 
Normalization improves convergence and stability.

## ✅ 4. Build CNN Model
Python

model = Sequential([
    Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    Conv2D(32, (3, 3)), Activation('relu'), MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), kernel_initializer='he_uniform'), Activation('relu'), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), kernel_initializer='he_uniform'), Activation('relu'), MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64), Activation('relu'), Dropout(0.5),
    Dense(1), Activation('sigmoid')
])

### purpose: 

Three convolutional blocks extract spatial features.

Dropout combats overfitting.

Sigmoid used for binary output.

## ✅ 5. Compile Model
Python

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### purpose:  
Binary classification task requires binary loss. 
Adam optimizer accelerates convergence.

## ✅ 6. Train Model
Python

history = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=10,
    validation_split=0.1
)

### purpose: 

Used validation_split to monitor overfitting.
Batch size optimized for GPU memory.

## ✅ 7. Model Evaluation
Python

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

Result: Achieved ~98% test accuracy, confirming generalization.

## ✅ 8. Model Saving & Reloading
Python

model.save("brain_tumor_cnn_model.keras")
model = load_model("brain_tumor_cnn_model.keras")

### purpose: 
Saved final model for reuse in deployment/inference.

## ✅ 9.Overfitting Detection
Python

import matplotlib.pyplot as plt
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title("Accuracy")

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")


### purpose: 
Captured training/validation loss and accuracy per epoch.
Plotted using Matplotlib.
Observation: Slight overfitting after epoch 6+ → prompted further regularization strategies.

## ✅ 10.Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# Re-train using augmented batches
model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=10, validation_data=(x_test, y_test))


### purpose: 
Combat overfitting and improve generalization.

## ✅ 11. Confusion Matrix & Classification Metrics

from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

### purpose: 
Beyond accuracy, precision/recall ensure robustness in real-world diagnostics.

## ✅ 12. Model Inference Pipeline

def predict_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    prediction = model.predict(img_input)[0][0]
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"🧠 Prediction: {label} | ✅ Confidence: {confidence:.2f}")

### Modularized inference step to test on any real-world image.

## 📈 Final Results

| Metric              | Value   |
| ------------------- | ------- |
| Train Accuracy      | \~97.5% |
| Val Accuracy        | \~99.5% |
| Test Accuracy       | \~98.2% |
| F1-Score (Tumor)    | 0.99    |
| F1-Score (No Tumor) | 0.97    |


## 📄 License
This project is licensed under the MIT License 

## 🙋‍♂️ Contact
📧 Gmail:[jayeshpardeshi161@gmail.com]
📌 LinkedIn:[] 📌 Portfolio:[]


