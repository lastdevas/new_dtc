import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the object detection model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

# Function to perform object detection on a single image
def detect_objects(image_tensor):
    # Perform inference on the image
    detections = detector(image_tensor)

    # Extract bounding boxes, classes, and scores
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    return boxes, classes, scores

def preprocess_image(uploaded_file):
    # Convert the uploaded image to a format that the model expects
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to the expected size
    resized_image = image.resize((1024, 1024))

    # Convert the resized image to a numpy array
    image_array = np.array(resized_image)

    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Add an extra dimension to the array to represent the batch size
    image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)

    return image_tensor

def main():
    st.title("Object Detection with TensorFlow Hub Faster R-CNN")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert and preprocess the uploaded image
        image_tensor = preprocess_image(uploaded_file)

        # Perform object detection
        boxes, classes, scores = detect_objects(image_tensor)

        # Display the results
        st.write("Detected Objects:")
        for box, class_id, score in zip(boxes, classes, scores):
            st.write(f"Class ID: {class_id}, Score: {score:.2%}")
            st.write(f"Bounding Box: {box}")

if __name__ == "__main__":
    main()
