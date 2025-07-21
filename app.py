import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model

# Load model once at start
@st.cache_resource
def load_brain_tumor_model():
    return load_model('model/brain_tumor_cnn_model.keras')

model = load_brain_tumor_model()

def preprocess_image(image, target_size=(64, 64)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return img_array.reshape(1, target_size[0], target_size[1], 3)

def predict_image(image):
    processed = preprocess_image(image)
    pred_prob = model.predict(processed)[0][0]
    label = 'Tumor' if pred_prob > 0.5 else 'No Tumor'
    return label, float(pred_prob)

# Streamlit UI
st.title("Brain Tumor Detection App - Batch Prediction")

uploaded_files = st.file_uploader(
    "Upload MRI images (jpg, jpeg, png)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        label, confidence = predict_image(image)
        results.append({
            'Filename': uploaded_file.name,
            'Prediction': label,
            'Confidence': f"{confidence:.4f}"
        })
        st.image(image, caption=f"{uploaded_file.name} - Prediction: {label} ({confidence:.2f})", width=200)

    df = pd.DataFrame(results)
    st.subheader("Prediction Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='brain_tumor_predictions.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload one or more MRI images to get predictions.")
