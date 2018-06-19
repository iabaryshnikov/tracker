import numpy as np
import math
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from onvif import ONVIFCamera
from time import sleep
from threading import Thread

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util



MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def speed_up():

  ptz.ContinuousMove(request)

def mov_to_face(ptz, request, x, y, to_x, to_y, speed_kof = 1, timeout=0, lenght = 700.0, width = 393.0):
  '''
  if x != -1 or y != -1:
    if (x < to_x +40 and x > to_x -40 and y < to_y +40 and y > to_y -40):
      request.Velocity.PanTilt._x = 0
      request.Velocity.PanTilt._y = 0
      ptz.ContinuousMove(request)
    else: 
      len_x = -(to_x - x)
      len_y = (to_y - y)
      vec = math.sqrt(len_x**2+len_y**2)
      vec_x = (len_x/(vec/100.0))/100.0
      vec_y = (len_y/(vec/100.0))/100.0
      print str(vec_x)+" : "+str(vec_y)
      request.Velocity.PanTilt._x = vec_x*speed_kof
      request.Velocity.PanTilt._y = vec_y*speed_kof
      ptz.ContinuousMove(request)
    sleep(timeout)
  '''
  if x != -1 or y != -1:
    if (x < to_x +40 and x > to_x -40 and y < to_y +40 and y > to_y -40):
      request.Velocity.PanTilt._x = 0
      request.Velocity.PanTilt._y = 0
      ptz.ContinuousMove(request)
    else:
      onevec = math.sqrt(lenght**2 + width**2)
      len_x = -(to_x - x)
      len_y = (to_y - y)
      vec = math.sqrt(len_x**2+len_y**2)
      speed = vec/onevec
      speed = int(speed*100)/100.0
      speed *= speed_kof
      print "speed - ", str(speed)
      vec_x = len_x/lenght
      vec_x = int(vec_x*100)/100.0
      vec_x *= speed_kof
      vec_y = len_y/width
      vec_y = int(vec_y*100)/100.0
      vec_y *= speed_kof
      print str(vec_y), " : ", str(vec_x)
      request.Velocity.PanTilt._x = vec_x
      request.Velocity.PanTilt._y = vec_y
      ptz.ContinuousMove(request)

  else:
    request.Velocity.PanTilt._x = 0
    request.Velocity.PanTilt._y = 0
    ptz.ContinuousMove(request)


print 'conection with camera...'
cap = VideoStream(src='rtsp://192.168.1.102:554/Streaming/Channels/101').start()
#cap = VideoStream(src=0).start()
mycam = ONVIFCamera('192.168.1.102', 80, 'admin', 'Supervisor', '/etc/onvif/wsdl/')

#mycam = ONVIFCamera('172.16.83.102:554', 80, 'admin', 'Supervisor')
#mycam = ONVIFCamera('192.168.13.12', 80, 'admin', 'Supervisor')
media = mycam.create_media_service()
profile = media.GetProfiles()[0]
ptz = mycam.create_ptz_service()
request = ptz.create_type('GetConfigurationOptions')
request.ConfigurationToken = profile.PTZConfiguration._token
ptz_configuration_options = ptz.GetConfigurationOptions(request)
request = ptz.create_type('ContinuousMove')
request.ProfileToken = profile._token
print 'sucsess conection.'


goal_person = ['person',-1,-1,-1]
lenght = 700
width = 393
speed = 0.1

fps = FPS().start()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      print '--'
      image_np = cap.read()
      image_np = cv2.resize(image_np, (lenght,width))


      cv2.rectangle(image_np,(lenght/3 - 30, width/3 -30),
        (lenght/3 + 30,width/3 + 30),(100,100,100),2)
      cv2.rectangle(image_np,(2*lenght/3 - 30, width/3 -30),
        (2*lenght/3 + 30,width/3 + 30),(100,100,100),2)
      cv2.rectangle(image_np,(lenght/3 - 30, 2*width/3 -30),
        (lenght/3 + 30,2*width/3 + 30),(100,100,100),2)
      cv2.rectangle(image_np,(2*lenght/3 - 30, 2*width/3 -30),
        (2*lenght/3 + 30,2*width/3 + 30),(100,100,100),2)

      
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')


      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})


      final_list = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      goal_person = ['person',-1,-1,-1]
      for i in range(len(final_list)):
        final_list[i][2] *= lenght
        final_list[i][3] *= width
        final_list[i][4] *= lenght
        final_list[i][5] *= width
        if final_list[i][0] == 'person' and goal_person[1] < final_list[i][1]:
          goal_person[1] = final_list[i][1]
          goal_person[2] = int(abs(final_list[i][2] - final_list[i][4])/2.0 + final_list[i][4])
          goal_person[3] = int(final_list[i][5])
      print goal_person

      '''if goal_person[2] == -1:
        speed = 0.1 
      elif speed == 0.1:
        len_x = (lenght/3 - goal_person[2])
        len_y = (width/3 - goal_person[3])
        vec = math.sqrt(len_x**2+len_y**2)
        speed += 0.1
        
      else:
        len_x = (lenght/3 - goal_person[2])
        len_y = (width/3 - goal_person[3])
        vec2 = math.sqrt(len_x**2+len_y**2)
        if speed < 0.5 and vec2 > vec/2.0:
          speed += 0.1
        if speed >= 0.2 and vec2 <= vec/2.0:
          speed -= 0.1
        print "vec - ",str(vec)
        print "vec2 - ",str(vec2)
      print speed'''
      mov_to_face(ptz, request, goal_person[2], goal_person[3], lenght/3, width/3, speed_kof=2)


      cv2.imshow('object detection', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        fps.stop()


        ptz.Stop({'ProfileToken': request.ProfileToken})


        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        break
      fps.update()