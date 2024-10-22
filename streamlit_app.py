import streamlit as st
import torch
import cv2
import numpy as np
from gtts import gTTS
import os

# Load the pre-trained YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8s')

import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize text-to-speech engine (you can replace this with LSTM-based TTS if needed)
engine = pyttsx3.init()

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')

# Start video capture from laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Function to convert object detection results into speech
def object_to_speech(detections):
    detected_objects = []
    
    # Loop through the detected objects
    for det in detections:
        # Get the label of the detected object (e.g., 'person', 'chair')
        object_name = model.names[int(det.cls)]
        
        # Add the object to the list if not already announced
        if object_name not in detected_objects:
            detected_objects.append(object_name)
            # Convert the object name to speech
            sentence = f"There is a {object_name}"
            print(sentence)  # Print the sentence for debugging
            engine.say(sentence)
    
    # Speak out all detected objects
    engine.runAndWait()

# Continuously capture frames from the camera and perform object detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    # Get detected objects
    detections = results[0].boxes  # This gives us the detected bounding boxes and labels
    
    # Convert detection results to speech
    object_to_speech(detections)
    
    # Draw the detected bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

