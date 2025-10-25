import streamlit as st
import cv2
import numpy as np
import easyocr
import time

st.title("Real-Time OCR with EasyOCR")

# Select languages
languages = st.multiselect(
    "Select OCR languages",
    ['en', 'hi'],
    default=['en', 'hi']
)

# GPU toggle
use_gpu = st.checkbox("Use GPU (if available)", value=True)

# Initialize OCR reader
reader = easyocr.Reader(languages, gpu=use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Store detected texts to avoid repetition
detected_texts = {}
cooldown_time = 3  # seconds before text can be recognized again

# Streamlit image placeholder
frame_placeholder = st.empty()

st.text("Press 'Stop' button to exit")

# Stop button
stop = st.button("Stop")

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture from webcam")
        break

    current_time = time.time()
    
    # Run OCR
    results = reader.readtext(frame)

    for bbox, text, conf in results:
        text_norm = text.strip().lower()

        # Skip if text is recently detected
        if text_norm in detected_texts and current_time - detected_texts[text_norm] < cooldown_time:
            continue

        detected_texts[text_norm] = current_time

        # Draw bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Remove old texts
    detected_texts = {t: ts for t, ts in detected_texts.items() if current_time - ts < cooldown_time}

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    if stop:
        break

# Release resources
cap.release()
st.text("Webcam stopped")
