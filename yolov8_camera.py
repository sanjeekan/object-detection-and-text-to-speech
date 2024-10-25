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
from ultralytics import YOLO
import pyttsx3

# Initialize text-to-speech engine (pyttsx3)
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
    
    for det in detections:
        object_name = model.names[int(det.cls)]
        
        if object_name not in detected_objects:
            detected_objects.append(object_name)
            sentence = f"There is a {object_name}"
            print(sentence)  # Print the sentence for debugging
            
            # Say the sentence without reinitializing or overlapping
            engine.say(sentence)
    
    # Run and wait only once per detection cycle
    engine.runAndWait()
    engine.stop()

# Continuously capture frames from the camera and perform object detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    # Get detected objects
    detections = results[0].boxes
    
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

# Stop the engine after exiting the loop
engine.stop()






