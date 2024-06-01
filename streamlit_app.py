import streamlit as st
import numpy as np
from collections import defaultdict, deque
#from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import tempfile
import cv2
import pandas as pd

# Streamlit app layout
st.title('YOLO Video Processing Dashboard')
st.sidebar.header('User Inputs')

# User inputs for model and video
model_name = st.sidebar.text_input('Model Name', 'yolov8n.pt')
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.20)
iou_threshold = st.sidebar.slider('IOU Threshold', 0.0, 1.0, 0.5)
selected_classes = st.sidebar.multiselect('Select Classes', options=list(range(80)), default=[0, 1, 2])

# Option to upload a video file or use webcam
use_webcam = st.sidebar.checkbox('Use Webcam')

uploaded_file = None
if not use_webcam:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None or use_webcam:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            source_video_path = temp_file.name
    else:
        source_video_path = None

    target_video_path = 'processed_video.mp4'

    # Load YOLO model
    model = YOLO(model_name)

    # Create BYTETracker instance
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    # Set default values for thickness and text scale
    thickness = 2  # Default line thickness
    text_scale = 0.5  # Smaller text scale

    # Create annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=2 * 30)  # Assume 30 fps
    line_zone_annotator = sv.LineZoneAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)

    # Define coordinates for speed calculation
    coordinates = defaultdict(lambda: deque(maxlen=30))  # Assume 30 fps

    # Data for traffic analysis report
    class_counts = defaultdict(int)
    speeds = []
    accelerations = []

    # Define callback function for video processing
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Model prediction on a single frame and conversion to supervision Detections
        results = model(frame, imgsz=640, verbose=False)[0]
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
            class_counts[class_name] += 1
            if len(coordinates[tracker_id]) < 30 / 2:
                labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / 30
                speed = distance / time * 3.6  # Convert to km/h
                speeds.append(speed)
                labels.append(f"#{tracker_id} {class_name} {confidence:.2f} {int(speed)} km/h")

                if len(coordinates[tracker_id]) > 1:
                    acceleration = (speeds[-1] - speeds[-2]) / time
                    accelerations.append(acceleration)

        # Annotate frame with traces, bounding boxes, and labels
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame

    # Process video frames
    if use_webcam:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = callback(frame, 0)
            st.image(frame, channels="BGR")
        cap.release()
    else:
        # Process the whole video and save it
        sv.process_video(
            source_path=source_video_path,
            target_path=target_video_path,
            callback=callback
        )
        st.video(target_video_path)

    # Display traffic analysis report
    total_vehicles = sum(class_counts.values())
    avg_speed = np.mean(speeds) if speeds else 0
    median_speed = np.median(speeds) if speeds else 0
    avg_acceleration = np.mean(accelerations) if accelerations else 0

    st.sidebar.subheader('Traffic Analysis Report')
    st.sidebar.write(f"Total Number of Vehicles: {total_vehicles}")
    for class_name, count in class_counts.items():
        st.sidebar.write(f"Total Number of {class_name.capitalize()}s: {count}")
    st.sidebar.write(f"Average Speed: {avg_speed:.2f} km/h")
    st.sidebar.write(f"Median Speed: {median_speed:.2f} km/h")
    st.sidebar.write(f"Average Acceleration: {avg_acceleration:.2f} m/s²")