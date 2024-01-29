# dtc_streamlit.py

import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import requests
from io import BytesIO

def load_image(image_path):
    return Image.open(image_path)

def predict_image(model, image):
    # Preprocess the image
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    label_id = tf.argmax(predictions[0]).numpy()

    # Get the class labels from the ImageNet class labels file
    label_names_file = tfds.load("imagenet2012", split="train", download=True).info.features["label"].names_file
    with tf.io.gfile.GFile(label_names_file) as f:
        label_names = f.read().splitlines()

    return label_names[label_id]

def main():
    st.title("ImageNet V2 Streamlit App")

    # Load pre-trained MobileNetV2 model
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load and preprocess the uploaded image
        image = load_image(uploaded_file)
        prediction = predict_image(model, image)

        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
