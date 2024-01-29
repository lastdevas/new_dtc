import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the image detector model from TensorFlow Hub
detector_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
detector = hub.load(detector_url)

# Function to apply image detector on a single image
def detect_objects(image_tensor):
    detector_output = detector(image_tensor)
    return detector_output

def main():
    st.title("Object Detection with TensorFlow Hub")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert the uploaded image to a format that the model expects
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        image_tensor = tf.convert_to_tensor([image_array])

        # Detect objects in the image
        detection_result = detect_objects(image_tensor)

        # Display the class names and probabilities
        class_names = detection_result['classification']
        probabilities = detection_result['probabilities']
        for name, prob in zip(class_names, probabilities):
            st.write(f"{name}: {prob:.2%}")

if __name__ == "__main__":
    main()
