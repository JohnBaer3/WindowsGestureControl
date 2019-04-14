######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from pynput.keyboard import Key, Controller
import webbrowser
import array as arr
import os

#webbrowser.open('http://python.org')



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 5

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

keyboard = Controller()
np.set_printoptions(threshold=np.nan)

bla = ''
timey = 0




def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y
#the y value for the rectangles that will be drawn
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)
#the x values for the rectangles that will be drawn
    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

#second position next to the points we made, connect these to draw
#the rectangle
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

#Draw the rectangles. Pass in frame, x position, y position, x2 position
#, y2 position, and color. 
    for i in range(9):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)
    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #The place of interest... we take the colors we took from the 
    #3X3 frame in the last step and make an image of size 90X10 with
    #3 color channels. 
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    
    for i in range(9):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    

#Returns filtered image from hand_histogram
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    #thresh = cv.merge((thresh, thresh, thresh))
    #return cv.bitwise_and(frame, thresh)
    return thresh

def draw_circles(frame, points):
    for point in points:
        cv2.circle(frame, point, 5, [0,255,255], -1)




'''keyboard.press(Key.ctrl)
keyboard.press('t')
keyboard.release(Key.ctrl)
keyboard.release('t')'''


'''while(True):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    frame = draw_rect(frame)
    histogram = hand_histogram(frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
box2 = [0,0,0,0]
'''

countey = 0
countey2 = 0
nothingCount = 0
commands = []



while(True):    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})



    # Draw the results of the detection (aka 'visulaize the results')

    
    frame, detected, box2 = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)


    width = 1280
    height = 720

    #frame = draw_rect(frame, box2[0]*height, box2[1]*width, box2[2]*height, box2[3]*width)
    #cv2.imshow('huh', frame)


    if detected == 'up':
        keyboard.press(Key.up)
        nothingCount = 0
    elif detected == 'down':
        keyboard.press(Key.down)
        nothingCount = 0
    elif detected == 'three':
        nothingCount = 0
        if len(commands) == 0:
            countey += 1
        elif commands[len(commands)-1] != 'three':
            countey += 1
        if countey > 2:
            commands.append('three')
            countey = 0
    elif detected == 'five':
        nothingCount = 0
        if len(commands) == 0:
            countey += 1
        elif commands[len(commands)-1] != 'five':
            countey += 1
        if countey > 2:
            commands.append('five')
            countey = 0
    elif detected == 'none':
        nothingCount = 0
        if len(commands) == 0:
            countey += 1
        elif commands[len(commands)-1] != 'none':
            countey += 1
        if countey > 2:
            commands.append('none')
            countey = 0
    else:
        nothingCount += 1
        if nothingCount == 4:
            commands = []

    i = 0
    if len(commands) > 1:
        if commands[0] == 'five':
            if commands[1] == 'three':
                webbrowser.open('https://old.reddit.com')
                commands = []
            elif commands[1] == 'none':
                webbrowser.open('https://twitter.com')
                commands = []
        elif commands[0] == 'three':
            if commands[1] == 'none':
                keyboard.press(Key.ctrl)
                keyboard.press('w')
                keyboard.release(Key.ctrl)
                keyboard.release('w')
                commands = []
            elif commands[1] == 'five':
                if(detected == 'five'):
                    if countey2 > 6:
                        keyboard.press(Key.ctrl)
                        keyboard.press(Key.tab)
                        keyboard.release(Key.ctrl)
                        keyboard.release(Key.tab)
                        countey2 = 0
                    else:
                        countey2+=1
        elif commands[0] == 'none':
            if commands[1] == 'three':
                webbrowser.open('https://www.youtube.com/watch?v=vNJpvahstFs&index=7&list=PLnGGovHKUZiCUfbwSyg5Dc_gTti6_ooWN&t=0s')
                commands = []
            elif commands[1] == 'five':
                webbrowser.open('https://www.youtube.com/watch?v=RNivb-ufiXY&index=15&list=PLnGGovHKUZiAjHsTrFRbcMB07DnzR6wdk')
                commands = []
                    

    print(commands)

    
#webbrowser.open('http://python.org')
#

    
    '''
        #Do the hand histogram thing here
        #Unless I can think of a better way to get the tip of a finger
        hist_mask_image = hist_masking(frame, histogram)
        #cv2.circle(hist_mask_image, (100,00), 5, [0,255,255], -1)
        cv2.imshow('huh', hist_mask_image)

        cv2.rectangle(hist_mask_image, (int(width * box2[1]), int(height * box2[0])), (int(width * box2[3]), int(height * box2[2])), (0, 255, 255), 10)
        rectangle = np.array([(box2[1]*width, box2[0]*height), (box2[1]*width,box2[2]*height), (box2[3]*width, box2[2]*height), (box2[3]*width,box2[0]*height)])
        mask = np.zeros_like(hist_mask_image)
        cv2.fillPoly(mask, np.int32([rectangle]), 255)
        masked = cv2.bitwise_and(hist_mask_image, mask)
    
        #masked is black and white image of only the finger
        
        _, contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                far_point = farthest_point(defects, )
                
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(frame, start, end, [0,255,0], 2)
                cv2.circle(frame, far, 5, [0,0,255], -1)
                traverse_point.append(far)'''
    

    # All the results have been drawn on the frame, so it's time to display it.
        
    cv2.imshow('huh', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break   




# Clean up
video.release()
cv2.destroyAllWindows()

