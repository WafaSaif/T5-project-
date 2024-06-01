import streamlit as st
import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import tempfile
import cv2
import os

# Set page title and configure page layout
st.set_page_config(page_title="Sahel - Riyadh Traffic Optimization Solution", page_icon="🚗", layout="wide")

# Add logo
#st.image("path_to_your_logo.png", use_column_width=True)  # Replace "path_to_your_logo.png" with the path to your logo image file

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

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create temporary file to save uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    SOURCE_VIDEO_PATH = tfile.name
    TARGET_VIDEO_PATH = 'target_video_1.mp4'

    # Create YOLO model instance
    model = YOLO(MODEL_NAME)

    # Create BYTETracker instance
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    # Create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # Create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Define line for LineZone
    LINE_START = sv.Point(50, 1500)
    LINE_END = sv.Point(3840 - 50, 1500)

    # Create LineZone instance
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

    # Create annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)

    # Define coordinates for speed calculation
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Define callback function for video processing
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
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
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Convert to km/h
                labels.append(f"#{tracker_id} {class_name} {confidence:.2f} {int(speed)} km/h")

        # Annotate frame with traces, bounding boxes, and labels
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Update line zone with current detections
        line_zone.trigger(detections)

        # Annotate frame with line zone results and return
        return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Process the video and save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=temp_output.name,
            callback=callback
        )

        # Display the processed video
        st.video(temp_output.name)

        # Option to download the processed video
        with open(temp_output.name, "rb") as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    # Clean up temporary file
    os.remove(SOURCE_VIDEO_PATH)
