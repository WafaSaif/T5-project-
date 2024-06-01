import streamlit as st
import numpy as np
from collections import defaultdict, deque
import pafy
import cv2
import tempfile
import os
from ultralytics import YOLO
import supervision as sv

# Set page title and configure page layout
st.set_page_config(page_title="Sahel - Riyadh Traffic Optimization Solution", page_icon="🚗", layout="wide")

# Customize the title
st.title("Sahel - Riyadh Traffic Optimization Solution")

# Set background color to green and font color to white
st.markdown(
    """
    <style>
    .css-1vcmnkv { 
        color: white; 
        background-color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define constants and parameters
MODEL_NAME = 'yolov8n.pt'  # Replace with the latest YOLOv8 model
MODEL_RESOLUTION = 640  # Resolution for model inference
CONFIDENCE_THRESHOLD = 0.20  # Adjust this threshold as needed
IOU_THRESHOLD = 0.5
selected_classes = [0, 1, 2]  # Replace with your desired class IDs

# Set default values for thickness and text scale
thickness = 2  # Default line thickness
text_scale = 0.5  # Smaller text scale

# YouTube live stream URL
stream_url = 'https://youtu.be/b7lsZ-0KiJw'

# Get video URL from YouTube live stream
video_pafy = pafy.new(stream_url)
best = video_pafy.getbest(preftype="mp4")

# Create YOLO model instance
model = YOLO(MODEL_NAME)

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create VideoCapture instance for live stream
cap = cv2.VideoCapture(best.url)

# Create annotators
box_annotator = sv.BoxAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=30 * 2)  # Assuming 30 fps
line_zone_annotator = sv.LineZoneAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)

# Define coordinates for speed calculation
coordinates = defaultdict(lambda: deque(maxlen=30))

# Define a line for LineZone (if needed)
LINE_START = sv.Point(50, 150)
LINE_END = sv.Point(1280 - 50, 150)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Model prediction on a single frame and conversion to supervision Detections
    results = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Only consider class IDs from selected_classes
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # Tracking detections
    detections = byte_tracker.update_with_detections(detections)

    # Store detection coordinates for speed calculation
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    # Generate labels and calculate speed if enough data is available
    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        class_name = model.model.names[class_id]
        if len(coordinates[tracker_id]) < 15:  # Using half of 30 fps for speed calculation
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / 30  # Assuming 30 fps
            speed = distance / time * 3.6  # Convert to km/h
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f} {int(speed)} km/h")

    # Annotate frame with traces, bounding boxes, and labels
    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Update line zone with current detections
    line_zone.trigger(detections)

    # Annotate frame with line zone results
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Display the annotated frame
    stframe.image(annotated_frame, channels="BGR", use_column_width=True)

cap.release()
