import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb

# Set the matplotlib backend to 'Agg' to avoid tkinter issues
plt.switch_backend('Agg')

# ResNet50+ - TensorFlow Model Loading
model = tf.keras.models.load_model('model_resnet50.h5')
explainer = lime_image.LimeImageExplainer()

def generate_lime_explanation(model, image):
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    lime_explanation, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_explanation_rgb = gray2rgb(lime_explanation)
    marked_explanation = mark_boundaries(image[0], mask)
    return marked_explanation

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
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
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].imshow(img[0])
ax[0].set_title(f'Original Image')

ax[1].imshow(lime_explanation)
ax[1].set_title(f'LIME Explanation : {predicted_class}')

# Save the figure
plt.savefig('lime_explanation.png')
plt.close()
