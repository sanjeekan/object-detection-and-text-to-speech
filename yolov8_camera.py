# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 pre-trained model (change the path to the model if necessary)
# model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with 'yolov8s.pt', 'yolov8m.pt', etc.

# # Start video capture from laptop camera
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video stream from camera.")
#     exit()

# # Continuously capture frames from the camera and perform object detection
# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Error: Failed to capture frame from camera.")
#         break
    
#     # Perform object detection on the frame
#     results = model(frame)
    
#     # Draw the detected bounding boxes and labels on the frame
#     annotated_frame = results[0].plot()

#     # Display the annotated frame
#     cv2.imshow('YOLOv8 Object Detection', annotated_frame)

#     # Press 'q' to quit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import pyttsx3
from ultralytics import YOLO
import time

# Initialize text-to-speech engine (you can replace this with LSTM-based TTS if needed)
engine = pyttsx3.init()

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')

# Start video capture from laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Set a timer to limit speech frequency
last_spoken_time = time.time()
speech_interval = 5  # Speak out detected objects every 5 seconds

# Function to convert object detection results into speech
def object_to_speech(detections):
    global last_spoken_time
    
    # Only speak if enough time has passed since the last spoken sentence
    if time.time() - last_spoken_time < speech_interval:
        return

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
    
    # Ensure no other loop is running and execute the text-to-speech process
    try:
        engine.runAndWait()
    except RuntimeError:
        print("Speech engine is already running, skipping.")

    # Update the last spoken time to prevent continuous speaking
    last_spoken_time = time.time()

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





