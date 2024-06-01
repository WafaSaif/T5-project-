import numpy as np
from collections import defaultdict, deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import supervision as sv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors

# Define constants and parameters
MODEL_NAME = 'yolov8n.pt'
MODEL_RESOLUTION = 640
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
selected_classes = [0, 1, 2]

LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)

model = load_model()

# Initialize BYTETracker
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Create LineZone instance
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Set default values for thickness and text scale
thickness = 2
text_scale = 0.5

# Create annotators
box_annotator = sv.BoxAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)
trace_annotator = sv.TraceAnnotator(thickness=thickness)
line_zone_annotator = sv.LineZoneAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)

# Define coordinates for speed calculation
coordinates = defaultdict(lambda: deque(maxlen=30))

def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame, imgsz=MODEL_RESOLUTION, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = byte_tracker.update_with_detections(detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        class_name = model.model.names[class_id]
        if len(coordinates[tracker_id]) < 15:
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / 30.0
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {class_name} {confidence:.2f} {int(speed)} km/h")

    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = process_frame(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Traffic Analysis using YOLO and Streamlit")
    st.sidebar.title("Settings")
    st.sidebar.slider('Confidence Threshold', 0.0, 1.0, CONFIDENCE_THRESHOLD, step=0.05)
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.write("Traffic Analysis Report")

    if st.button("Generate Report"):
        st.write(f"Total number of vehicles: {line_zone.count()}")
        vehicle_count = defaultdict(int)
        for track_id in coordinates:
            class_id = byte_tracker[track_id].class_id
            vehicle_count[model.model.names[class_id]] += 1

        for vehicle_class, count in vehicle_count.items():
            st.write(f"Total number of {vehicle_class}: {count}")

        speeds = [abs(coordinates[track_id][-1] - coordinates[track_id][0]) / (len(coordinates[track_id]) / 30.0) * 3.6 for track_id in coordinates if len(coordinates[track_id]) >= 15]
        if speeds:
            st.write(f"Average speed: {np.mean(speeds):.2f} km/h")
            st.write(f"Median speed: {np.median(speeds):.2f} km/h")
            accelerations = [(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
            if accelerations:
                st.write(f"Average acceleration: {np.mean(accelerations):.2f} km/h²")
        else:
            st.write("No speed data available")

if __name__ == "__main__":
    main()
