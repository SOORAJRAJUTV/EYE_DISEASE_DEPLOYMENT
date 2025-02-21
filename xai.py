import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
from lime import lime_image
from flask import current_app as app  # Import Flask app context

# Function to load the model safely from UPLOAD_FOLDER
def load_model():
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model_resnet50.h5')  # Load from uploads folder
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file {model_path} not found!")
    
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess Image for Model Input
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Resize image to match model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255  # Normalize pixel values
    return img

# Generate LIME Explanation
def generate_lime_explanation(model, image, explainer):
    explanation = explainer.explain_instance(
        image[0].astype('double'), model.predict, 
        top_labels=1, hide_color=0, num_samples=1000
    )
    lime_explanation, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, 
        num_features=5, hide_rest=False
    )
    marked_explanation = mark_boundaries(image[0], mask)
    return marked_explanation

