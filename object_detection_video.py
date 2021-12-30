import tensorflow as tf
import os
import pathlib

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

import cv2 

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import time

# 내 로컬에 설치된 레이블 파일을, 인덱스와 연결시킨다.
PATH_TO_LABELS = 'C:\\Users\\pys1\\Documents\\GitHub\\openCV\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

# 모델 로드하는 함수.

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# 위의 사이트에서 모델을 가져올수있다.
#http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# /20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"

    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)


def show_inference(detection_model, image_np) :
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detection_model(input_tensor)
#     print(detections)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    print(detections)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow( 'result' , image_np_with_detections)

def save_inference(detection_model, image_np, video_writer):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detection_model(input_tensor)
#     print(detections)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    print(detections)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    video_writer.write(image_np_with_detections)




# 비디오를 실행하는 코드
cap = cv2.VideoCapture('data/video.mp4')

if cap.isOpened() == False:
    print('비디오 실행 에러')
else :
    frame_width = int( cap.get(3) )
    frame_height = int( cap.get(4) )
    out = cv2.VideoWriter('data/output.avi', 
            cv2.VideoWriter_fourcc('M','J','P',"G") ,
            20, 
            (frame_width,frame_height) )

    
    # 비디오 캡쳐에서, 이미지를 1장씩 가져온다.
    # 이 1장의 이미지를, 오브젝트 디텍션 한다.
    while cap.isOpened() :
        ret, frame = cap.read()

        if ret == True:
            # frame 이 이미지에 대한 넘파이 어레이 이므로
            # 이 frame 을 오브젝트 디텍션 한다.

            # 학습용으로, 동영상으로 저장하는 코드롤
            # 수정하세요.

            start_time = time.time()
            # save_inference(detection_model, frame, out)
            
            # 동영상을 실시간으로 화면에서 디텍팅하는것
            show_inference(detection_model, frame)

            end_time = time.time()

            print('연산에 걸린 시간', str(end_time-start_time))

            if cv2.waitKey(27) & 0xFF == 27: 
                break
        else : 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()