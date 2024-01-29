import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import gdown
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")

        # Create and place the "Choose an image" button
        choose_button = tk.Button(root, text="Choose an image...", command=self.choose_image)
        choose_button.pack(pady=10)

        # Create and place a label for drag and drop instructions
        drag_label = tk.Label(root, text="Drag and drop file here", relief="solid", width=40, height=5, bd=2)
        drag_label.pack(pady=10)

        # Bind drag and drop events to the label
        drag_label.drop_target_register(tk.DND_FILES)
        drag_label.dnd_bind('<<Drop>>', self.drop_event)

        # Create and place a label for displaying the result image
        self.result_label = tk.Label(root)
        self.result_label.pack()

        # Download the model file from Google Drive
        file_id = '1WMUbo1u8a5lwfuBSA2NOPCh58xQkaImh'
        output_file = 'saved_model.pb'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_file, quiet=False)

        # Load the saved model
        self.model = tf.saved_model.load(output_file)

        # Load label map (replace 'label_map.pbtxt' with your label map file)
        label_map_path = 'path/to/label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    def choose_image(self):
        # Open file dialog to choose an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        # Perform object detection and update the result image in the GUI
        self.display_result(file_path)

    def drop_event(self, event):
        # Get the dropped file path
        file_path = event.data

        # Perform object detection and update the result image in the GUI
        self.display_result(file_path)

    def display_result(self, image_path):
        # Load and preprocess the image
        img_array = self.load_and_preprocess_image(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)

        # Perform object detection
        detections = self.model(input_tensor)

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_array[0],
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.30,
            agnostic_mode=False
        )

        # Display the result image
        photo = ImageTk.PhotoImage(Image.fromarray(img_array))
        self.result_label.config(image=photo)
        self.result_label.image = photo

    def load_and_preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize image to fit the model's expected size
        img_array = np.array(img)
        return img_array

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
