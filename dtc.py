import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw

# Load the image detector model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

# Function to apply image detector on a single image
def detect_objects(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Detect objects in the image
    result = detector(tf.convert_to_tensor([image_np], dtype=tf.uint8))

    # Extract bounding boxes, class names, and scores
    boxes = result['detection_boxes'][0].numpy()
    classes = result['detection_classes'][0].numpy().astype(int)
    scores = result['detection_scores'][0].numpy()

    return boxes, classes, scores

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            box = [xmin * image.width, ymin * image.height, xmax * image.width, ymax * image.height]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{classes[i]}", fill="red", font=None)

    return image

def main():
    st.title("Object Detection with TensorFlow Hub")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect objects in the image
        boxes, classes, scores = detect_objects(image)

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image.copy(), boxes, classes, scores)

        # Display the image with bounding boxes
        st.image(image_with_boxes, caption="Object Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()
