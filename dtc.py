import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw
import time

# Load the image detector model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

# Function to load an image from path
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

# Function to run the detector on an image
def run_detector(detector, img):
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    st.write("Found %d objects." % len(result["detection_scores"]))
    st.write("Inference time: ", end_time - start_time)

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])

    st.image(image_with_boxes, caption="Object Detection Result", use_column_width=True)

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            box = [xmin * image.width, ymin * image.height, xmax * image.width, ymax * image.height]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{classes[i].decode('utf-8')}", fill="red", font=None)

    return image

def main():
    st.title("Object Detection with TensorFlow Hub")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load and run the detector on the image
        img = load_img(uploaded_file)
        run_detector(detector, img)

if __name__ == "__main__":
    main()
