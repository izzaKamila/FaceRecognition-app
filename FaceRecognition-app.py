import streamlit as st
from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n_custom1.pt")

# Set up Streamlit for the webcam input
st.title("Face Recognition with YOLOv8")
st.write("Press 'q' to stop the webcam.")

# Start the webcam capture
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    st.error("No Camera found.")
else:
    stframe = st.empty()  # This is the placeholder for webcam frames

    while True:
        ret, image = cam.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # Convert the image to RGB (Streamlit uses RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Measure the inference time
        start_time = time.time()
        result = model.predict(image_rgb, show=False)

        # Print the inference time to the console
        st.write("Inference Time: ", time.time() - start_time)

        # Get the result image
        annotated_image = result[0].plot()  # Annotated image with predictions

        # Display the result in Streamlit
        stframe.image(annotated_image, channels="RGB", use_column_width=True)

        # If the user presses 'q', exit the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the webcam and clean up
    cam.release()
    cv2.destroyAllWindows()
