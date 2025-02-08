<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb


# ResNet50+ - TensorFlow Model Loading

model = tf.keras.models.load_model('model_resnet50.h5')
explainer = lime_image.LimeImageExplainer()


def generate_lime_explanation(model, image):
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    lime_explanation, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # Convert Lime explanation to RGB format
    lime_explanation_rgb = gray2rgb(lime_explanation)

    # Overlay Lime explanation on the original image using the mask
    marked_explanation = mark_boundaries(image[0], mask)

    return marked_explanation

# Function to preprocess image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (256, 256))  # Resize to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255  # Normalize
    return img

# Provide the path to your image
image_path = r"C:\Users\soora\OneDrive\Desktop\WEBpro\static\4637_right.jpg"

# Read and preprocess the image
img = cv2.imread(image_path)
if img is None:
    print("Error: Unable to read the image.")
else:
    img = preprocess_image(img)

    # Make predictions
    class_labels = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
    class_index = np.argmax(model.predict(img))
    predicted_class = class_labels[class_index]

    print("Predicted class:", predicted_class)

lime_explanation = generate_lime_explanation(model, img)

# Plot results
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# ax[0].imshow(img[0])
# ax[0].set_title(f'Original Image')

# ax[1].imshow(lime_explanation)
# ax[1].set_title(f'LIME Explanation : {predicted_class}')

# plt.show()


fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].imshow(img[0])
ax[0].set_title(f'Original Image')

ax[1].imshow(lime_explanation)
ax[1].set_title(f'LIME Explanation : {predicted_class}')

# Save the figure
plt.savefig('lime_explanation.png')  # Change the filename as needed
plt.close()  # Close the figure to release memory
=======
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
>>>>>>> 925c8d5c66f6b1a42d4d630202a46df2ac618a99
