import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from xai import explain_image  # Import the XAI function

# Function to load the model
@st.cache_resource
def load_trained_model():
    return load_model("Eye-Disease-Detection/trained_resnet50_final.h5", compile=False)

# Load the model
model = load_trained_model()

# Function to preprocess image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (256, 256))  # Resize to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255  # Normalize to [0, 1]
    return img

# Class descriptions
class_descriptions = {
    "Cataract": "Cataract is a condition where the eye's lens becomes cloudy, causing blurry vision. "
                "The model detected opacity in the lens area, which is a key feature of cataracts.",
    
    "Diabetic Retinopathy": "Diabetic Retinopathy is caused by diabetes and leads to damage in retinal blood vessels. "
                            "The model identified abnormalities such as hemorrhages, microaneurysms, or exudates in the retina.",
    
    "Glaucoma": "Glaucoma is a disease that damages the optic nerve, often due to increased eye pressure. "
                "The model detected changes in the optic disc, such as an increased cup-to-disc ratio, which is a key indicator of glaucoma.",
    
    "Normal": "No abnormalities were detected. The eye appears healthy with a clear lens and a well-structured retina."
}

# Streamlit app
st.title("🩺 Eye Disease Detection")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and decode the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize image for display
    display_img = cv2.resize(img, (300, 300))  # Resize for better UI display

    # Display the uploaded image
    st.image(display_img, caption="📌 Uploaded Image", use_container_width=False, width=300)

    # Preprocess and predict
    img_processed = preprocess_image(img)
    class_labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
    prediction = model.predict(img_processed)
    predicted_class = class_labels[np.argmax(prediction)]

     # Display prediction result
    st.subheader(f"🎯 Predicted Class: **{predicted_class}**")

    # Explainability with LIME
    st.write("🔍 **Explainable AI (LIME)**")
    explained_img = explain_image(img, model)  # Pass the original image (3D array)
    explained_img_resized = cv2.resize(explained_img, (300, 300))  # Resize explanation for better fit
    st.image(explained_img_resized, caption="LIME Explanation", use_container_width=False, width=300)

    # Display explanation for the prediction
    st.write("🧐 **Why was this predicted?**")
    st.write(class_descriptions[predicted_class])
