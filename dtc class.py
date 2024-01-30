import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the image classification model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)

# Function to apply image classification on a single image
def classify_image(image_tensor):
    predictions = model(image_tensor)
    return predictions

def preprocess_image(uploaded_file):
    # Convert the uploaded image to a format that the model expects
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to the expected size
    resized_image = image.resize((224, 224))
    
    # Convert the resized image to a numpy array
    image_array = np.array(resized_image)
    
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Add an extra dimension to the array to represent the batch size
    image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)
    return image_tensor

def main():
    st.title("Engine Image Classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert and preprocess the uploaded image
        image_tensor = preprocess_image(uploaded_file)

        # Classify objects in the image
        predictions = classify_image(image_tensor)

        # Display the top predicted classes and probabilities
        top_classes = tf.argmax(predictions, axis=-1)
        top_probs = tf.reduce_max(predictions, axis=-1)
        
        st.write("Top Predictions:")
        for class_index, prob in zip(top_classes.numpy(), top_probs.numpy()):
            st.write(f"Class {class_index} with probability {prob:.2%}")

if __name__ == "__main__":
    main()