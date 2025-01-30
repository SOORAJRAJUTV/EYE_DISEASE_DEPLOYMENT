import numpy as np
import tensorflow as tf
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import cv2

# Function to preprocess the image for prediction
def preprocess_image(input_data):
    img_array = tf.image.resize(input_data, (256, 256))  # Resize image to (256, 256)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array.numpy()  # Convert tensor to numpy array

# Function to make predictions with the model
def model_predict(model, input_data):
    if isinstance(input_data, list):
        input_data = np.array(input_data)
    
    img_array = np.array([preprocess_image(img) for img in input_data])
    prediction = model.predict(img_array)
    
    return prediction

# Function to explain the image using LIME
def explain_image(img, model):
    if not isinstance(img, np.ndarray):
        img_processed = np.array(img)
    else:
        img_processed = img

    # Convert image to grayscale for better segmentation focus
    gray_img = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)

    # Find contours and apply bounding box to isolate eye region
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_processed = img_processed[y:y+h, x:x+w]  # Crop image to focus on the eye

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_processed,  
        lambda x: model_predict(model, x),
        top_labels=1,
        hide_color=None,  # Ensuring no external blank area is added
        num_samples=1000
    )

    # Get explanation visualization
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,  # Highlight only important areas
        num_features=8,  # Reduce superpixel count to prevent excessive overlay
        hide_rest=False
    )

    # Overlay explanation mask on the cropped eye image
    explained_img = mark_boundaries(temp, mask, color=(1, 0, 0))

    return explained_img
