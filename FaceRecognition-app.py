import streamlit as st
from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n_custom1.pt")

# Set up Streamlit UI
st.title("Face Recognition with YOLO")
st.write("Use your webcam or upload a video to perform face recognition.")

# Streamlit file uploader or camera input
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
use_webcam = st.checkbox("Use Webcam", False)

# Define webcam usage logic
if use_webcam:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("No Camera Found")
    else:
        while True:
            ret, image = cam.read()
            if not ret:
                break
            
            # Process frame with YOLO
            _time_mulai = time.time()
            result = model.predict(image, show=False)  # Don't use show=True because it's handled by Streamlit

            # Display the frame with detections
            frame = result[0].plot()  # Visualize results

            # Calculate time for each frame
            st.write(f"Processing Time: {time.time() - _time_mulai:.2f} seconds")
            
            # Convert frame to image to display on Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cam.release()

# Handling video file upload
elif video_file is not None:
    video_bytes = video_file.read()
    st.video(video_bytes)

else:
    st.warning("Please select a video or enable webcam input.")
