import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import streamlit as st
import tempfile

# Define flags
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

# Set page title and configure page layout
st.set_page_config(page_title="Sahel - Riyadh Traffic Optimization Solution", page_icon="🚗", layout="wide")
st.title("Sahel - Riyadh Traffic Optimization Solution")

# Sidebar settings
st.sidebar.title('Custom Object Detection')
use_webcam = st.sidebar.button('Use Webcam')
confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.3)
video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
custom_classes = st.sidebar.checkbox('Use Custom Classes')

class_names = utils.read_class_names(cfg.YOLO.CLASSES)
if custom_classes:
    assigned_class = st.sidebar.multiselect('Select The Custom Classes', list(class_names.values()), default='car')

stop_button = st.sidebar.button('Stop Processing')
if stop_button:
    st.stop()

save_video = st.button('Save Results')
stframe = st.empty()

# TensorFlow session configuration
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = FLAGS.size

# Load the YOLO model
if FLAGS.framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

# Define the main processing function
def process_frame(frame):
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if FLAGS.framework == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=confidence, input_shape=tf.constant([input_size, input_size]))
    else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=confidence
    )

    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    allowed_classes = assigned_class if custom_classes else list(class_names.values())
    
    if FLAGS.crop:
        crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, os.path.join(os.getcwd(), 'detections', 'crop'), allowed_classes)
    
    if FLAGS.count:
        counted_classes = count_objects(pred_bbox, by_class=False, allowed_classes=allowed_classes)
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
    else:
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
    
    return image

def main(_argv):
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file_buffer:
        vid = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO if not use_webcam else tfflie.name
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'VP08')
    out = cv2.VideoWriter('output.webm', codec, fps, (width, height))

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_frame(frame)
        result = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        if save_video:
            out.write(result)
        result = cv2.resize(result, (720, int(720 * height / width)))
        stframe.image(result, channels='BGR', use_column_width=True)

    st.success('Video saved')
    st.text('Video is Processed')
    output_vid = open('output.webm', 'rb')
    out_bytes = output_vid.read()
    st.text('OutPut Video')
    st.video(out_bytes)
    vid.release()
    out.release()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
