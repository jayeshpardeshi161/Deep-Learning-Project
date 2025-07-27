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

image_directory = 'datasets/'
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

What I did:
Loaded .jpg images from two folders: yes/ and no/.

Assigned labels: 1 for tumor, 0 for no tumor.

### Resizing all images to a uniform shape
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

### ✅ 3. Data Preprocessing

dataset = np.array(dataset)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

### Normalize pixel values [0-255] ➝ [0-1]
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

### purpose: 
Train/test split allows generalization testing. 
Normalization improves convergence and stability.

### ✅ 4. Build CNN Model

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

### ✅ 5. Compile Model


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### purpose:  
Binary classification task requires binary loss. 
Adam optimizer accelerates convergence.

### ✅ 6. Train Model


history = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=10,
    validation_split=0.1
)

### purpose: 

Used validation_split to monitor overfitting.
Batch size optimized for GPU memory.

### ✅ 7. Model Evaluation

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

## Result: Achieved ~98% test accuracy, confirming generalization.

### ✅ 8. Model Saving & Reloading

model.save("brain_tumor_cnn_model.keras")
model = load_model("brain_tumor_cnn_model.keras")

### purpose: 
Saved final model for reuse in deployment/inference.

### ✅ 9.Overfitting Detection

import matplotlib.pyplot as plt
### Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title("Accuracy")

### Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")


### purpose: 
Captured training/validation loss and accuracy per epoch.
Plotted using Matplotlib.
Observation: Slight overfitting after epoch 6+ → prompted further regularization strategies.

### ✅ 10.Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

### Re-train using augmented batches
model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=10, validation_data=(x_test, y_test))


### purpose: 
Combat overfitting and improve generalization.

### ✅ 11. Confusion Matrix & Classification Metrics

from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

### purpose: 
Beyond accuracy, precision/recall ensure robustness in real-world diagnostics.

### ✅ 12. Model Inference Pipeline

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

--

## ✅ What I Did

### Step 1: Import Required Libraries

I imported essential Python libraries required for building a Convolutional Neural Network (CNN) model for brain tumor detection using MRI image classification. 
These libraries serve different purposes such as image preprocessing, numerical computation, deep learning model creation, dataset splitting, and normalization. 
Ensuring the right set of libraries are imported at the beginning helps streamline the project and prevents redundant imports later in the pipeline.

| **Python Code**                                                                                 | **# Comments**                                                                                                                               |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `import os`                                                                                     | To access and manage file directories, useful for navigating through MRI image folders.                                                      |
| `import cv2`                                                                                    | Used for reading and preprocessing image files (e.g., resizing, grayscale conversion).                                                       |
| `import numpy as np`                                                                            | For performing efficient numerical and matrix operations on image data.                                                                      |
| `from PIL import Image`                                                                         | Provides additional image manipulation functionality, especially format conversion.                                                          |
| `import tensorflow as tf`                                                                       | Main deep learning framework used to build and train the CNN model.                                                                          |
| `from tensorflow.keras.utils import normalize`                                                  | To scale pixel values between 0 and 1, improving training efficiency.                                                                        |
| `from sklearn.model_selection import train_test_split`                                          | To divide the dataset into training and testing sets for model validation.                                                                   |
| `from tensorflow.keras.models import Sequential`                                                | Allows the creation of a simple sequential CNN architecture layer by layer.                                                                  |
| `from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense` | Key CNN components: convolution, pooling, activation functions, dropout for regularization, flattening, and dense layers for classification. |

### Step 2: Load Dataset

In this step, I verified the working directory to ensure that the dataset path was correctly set. 
This is crucial to prevent file-not-found errors when loading MRI image data. 
I used the os module to check and confirm that the current working directory matches the dataset location, which is essential for accessing and loading image files for further processing.

| **Python Code**                                                                           | **# Comments**                                                                        |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `import os`                                                                               | Used to interact with the file system and manage paths.                               |
| `print(os.getcwd())`                                                                      | Printed the current working directory to verify that it points to the dataset folder. |
| **Output:** `D:\Projects\Deep Learning Projects\Brain Tumor Detection Using CNN\datasets` | Confirms that the dataset directory is correctly set for loading images.              |

To begin processing the brain tumor MRI data, I ensured that the script pointed to the correct working directory. 
I verified the current directory using os.getcwd() and updated it using os.chdir() to match the dataset location. 
Then, I defined the dataset path and loaded the image file names from the respective folders: 'yes' (tumor present) and 'no' (no tumor). This setup is necessary before beginning the image preprocessing and labeling phase.

| **Python Code**                                                                                              | **# Comments**                                                           |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `import os`                                                                                                  | Module used for directory operations.                                    |
| `print("Current Directory:", os.getcwd())`                                                                   | Printed the current working directory to verify initial script location. |
| **Output:** `Current Directory: D:\Projects\Deep Learning Projects\Brain Tumor Detection Using CNN\datasets` | Confirms the initial directory before any change.                        |
| `os.chdir(r"D:\Projects\Deep Learning Projects\Brain Tumor Detection Using CNN\datasets")`                   | Changed the working directory to the dataset folder.                     |
| `print("Current Directory:", os.getcwd())`                                                                   | Verified the change to ensure the correct path is active.                |
| **Output:** `Current Directory: D:\Projects\Deep Learning Projects\Brain Tumor Detection Using CNN\datasets` | Confirms that directory change was successful.                           |
| `image_directory = r'D:\Projects\Deep Learning Projects\Brain Tumor Detection Using CNN\datasets\\'`         | Defined the base path for accessing the 'yes' and 'no' image folders.    |
| `no_tumor_images = os.listdir(image_directory + 'no/')`                                                      | Loaded file names of all MRI images labeled as "no tumor".               |
| `yes_tumor_images = os.listdir(image_directory + 'yes/')`                                                    | Loaded file names of all MRI images labeled as "tumor present".          |


### Step 3: Image Preprocessing and Label Assignment

In this step, I initialized empty lists to store image data and their corresponding labels. 
I then looped through all the MRI images in both the 'no' and 'yes' tumor directories. 
Each image was read using OpenCV, converted from BGR to RGB using Pillow (PIL), resized to a standard input size for the CNN, and then appended to the dataset as a NumPy array. 
Labels were assigned as 0 for no tumor and 1 for tumor present. I also ensured the dataset remained a Python list to avoid potential type issues during processing.

| **Python Code**                                                  | **# Comments**                                                             |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `dataset = []`<br>`labels = []`                                  | I reset both lists to store MRI image data and their corresponding labels. |
| `for image_name in no_tumor_images:`                             | I started looping through all the 'no tumor' images.                       |
| `if image_name.endswith('.jpg'):`                                | Ensured only JPEG images are processed.                                    |
| `img_path = os.path.join(image_directory, 'no', image_name)`     | Constructed the full path for each 'no tumor' image.                       |
| `img_cv = cv2.imread(img_path)`                                  | Loaded the image using OpenCV.                                             |
| `if img_cv is not None:`                                         | Checked for any corrupted or unreadable images.                            |
| `img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))` | Converted image from BGR to RGB using PIL.                                 |
| `img = img.resize((INPUT_SIZE, INPUT_SIZE))`                     | Resized the image to match the CNN input dimensions.                       |
| `dataset.append(np.array(img))`                                  | Added the processed image (as a NumPy array) to the dataset list.          |
| `labels.append(0)`                                               | Appended label `0` for images with no tumor.                               |
| `if not isinstance(dataset, list): dataset = list(dataset)`      | Ensured `dataset` remains a list before continuing.                        |
| `for image_name in yes_tumor_images:`                            | I then looped through all the 'yes tumor' images.                          |
| `if image_name.endswith('.jpg'):`                                | Processed only `.jpg` format files.                                        |
| `img_path = os.path.join(image_directory, 'yes', image_name)`    | Constructed path for each 'yes tumor' image.                               |
| `img_cv = cv2.imread(img_path)`                                  | Loaded each tumor image using OpenCV.                                      |
| `if img_cv is not None:`                                         | Skipped any unreadable files.                                              |
| `img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))` | Converted image to RGB format using Pillow.                                |
| `img = img.resize((INPUT_SIZE, INPUT_SIZE))`                     | Resized image to uniform input size for the CNN model.                     |
| `dataset.append(np.array(img))`                                  | Added the image array to the dataset.                                      |
| `labels.append(1)`                                               | Labeled tumor images with `1`.                                             |


### Step 4: Data Preprocessing

After collecting and labeling the MRI images, I converted the dataset and labels lists into NumPy arrays to optimize memory usage and computational speed. 
Then, I split the data into training and testing sets (80%-20%) using train_test_split. 
Finally, I normalized the pixel values to a 0–1 range using normalize() from Keras utilities, which helps the neural network converge faster and more accurately.

| **Python Code**                                                                                       | **# Comments**                                                                |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `dataset = np.array(dataset)`<br>`labels = np.array(labels)`                                          | Converted image and label lists into NumPy arrays for efficient processing.   |
| `x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)` | Split the data into training (80%) and testing (20%) sets.                    |
| `x_train = normalize(x_train, axis=1)`<br>`x_test = normalize(x_test, axis=1)`                        | Normalized pixel values across each image to bring them to a scale of 0 to 1. |


### Step 5: Build CNN Model

I created a CNN model using Keras’ Sequential API. I began with an Input layer defining the shape of the input images. 
The model architecture included three convolutional blocks with increasing filter sizes, each followed by ReLU activation and max pooling. 
After flattening the output, I added a dense layer with dropout to reduce overfitting, followed by the final output layer with sigmoid activation for binary classification.

| **Python Code**                                                                                                                                    | **# Comments**                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `from tensorflow.keras.layers import Input`                                                                                                        | Imported the `Input` layer to define input shape.                           |
| `model = Sequential()`                                                                                                                             | Initialized the Sequential model.                                           |
| `model.add(Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)))`                                                                                              | Defined the input shape for the CNN.                                        |
| `model.add(Conv2D(32, (3, 3)))`<br>`model.add(Activation('relu'))`<br>`model.add(MaxPooling2D(pool_size=(2, 2)))`                                  | First Conv block with 32 filters, ReLU activation, and 2x2 max pooling.     |
| `model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))`<br>`model.add(Activation('relu'))`<br>`model.add(MaxPooling2D(pool_size=(2, 2)))` | Second Conv block with same filters but He uniform initializer.             |
| `model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))`<br>`model.add(Activation('relu'))`<br>`model.add(MaxPooling2D(pool_size=(2, 2)))` | Third Conv block with 64 filters.                                           |
| `model.add(Flatten())`                                                                                                                             | Flattened the 2D outputs to 1D for Dense layers.                            |
| `model.add(Dense(64))`<br>`model.add(Activation('relu'))`<br>`model.add(Dropout(0.5))`                                                             | Added a dense hidden layer and applied dropout for regularization.          |
| `model.add(Dense(1))`<br>`model.add(Activation('sigmoid'))`                                                                                        | Final output layer with sigmoid for binary classification (tumor/no tumor). |

### Step 6: Compile the Model

I compiled the CNN model using the binary cross-entropy loss function, suitable for binary classification. 
I selected the Adam optimizer for its adaptive learning capabilities and specified accuracy as the performance metric to monitor during training.

| **Python Code**                                                                     | **# Comments**                                                            |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])` | Compiled the model with appropriate loss, optimizer, and accuracy metric. |






## 📄 License
This project is licensed under the MIT License 

## 🙋‍♂️ Contact
📧 Gmail:[jayeshpardeshi161@gmail.com]
📌 LinkedIn:[] 📌 Portfolio:[]


