import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the image detector model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key: value.numpy() for key, value in result.items()}

    st.image(img.numpy(), caption="Uploaded Image", use_column_width=True)

    st.subheader("Top Predictions:")
    for i in range(min(5, len(result["detection_scores"]))):
        st.write(
            f"Class {result['detection_class_entities'][i]} with probability "
            f"{result['detection_scores'][i]*100:.2f}%"
        )

if __name__ == "__main__":
    st.title("Object Detection with TensorFlow Hub and Streamlit")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Detect objects in the image
        run_detector(detector, uploaded_file)
