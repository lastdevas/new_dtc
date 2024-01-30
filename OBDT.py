import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the object detection model from TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
model = hub.load(model_url)

# Function to apply object detection on a single image
def detect_objects(image_tensor):
    detections = model(image_tensor)
    return detections

def preprocess_image(uploaded_file):
    # Convert the uploaded image to a format that the model expects
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to the expected size
    resized_image = image.resize((512, 512))
    
    # Convert the resized image to a numpy array
    image_array = np.array(resized_image)
    
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Add an extra dimension to the array to represent the batch size
    image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)
    return image_tensor

def main():
    st.title("Object Detection using EfficientDet")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert and preprocess the uploaded image
        image_tensor = preprocess_image(uploaded_file)

        # Detect objects in the image
        detections = detect_objects(image_tensor)

        # Display the number of detected objects
        num_objects = int(detections["num_detections"].numpy())
        st.write(f"Number of Objects Detected: {num_objects}")

        # Display details for each detected object
        st.write("Object Details:")
        for i in range(num_objects):
            class_name = detections["detection_classes"][0][i].numpy()
            class_score = detections["detection_scores"][0][i].numpy()
            st.write(f"Object {i + 1}: Class {class_name} with confidence {class_score:.2%}")

if __name__ == "__main__":
    main()
