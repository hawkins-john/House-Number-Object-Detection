"""
Multi-digit House Number Detection Using Convolutional Neural Networks
Author: John Hawkins
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import cv2
import numpy as np
from scipy.io import loadmat
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#tf.compat.v1.disable_eager_execution()


# reduce image to half its size
def reduce_image(image):
    temp_image = np.copy(image)

    # generate 5-tap filter
    vert_filter = np.array([[1/16.], [4/16.], [6/16.], [4/16.], [1/16.]])
    horiz_filter = np.array([1/16., 4/16., 6/16., 4/16., 1/16.]).reshape((1,5))
    filter = np.dot(vert_filter, horiz_filter)
    # convolve 5-tap filter with image
    filtered_image = cv2.filter2D(temp_image, ddepth=-1, kernel=filter, dst=None)
    # reduce shape of image by 1/2
    reduced_image = filtered_image[::2, ::2]

    return reduced_image

# create Guassian pyramid of image
def gaussian_pyramid(image, levels):
    temp_image = np.copy(image)

    gauss_pyramid = [temp_image]
    for i in range(levels-1):
        # reduce image
        temp_image = reduce_image(temp_image)
        # append reduced image to pyramid list
        gauss_pyramid.append(temp_image)

    return gauss_pyramid

def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


# define model path
file_path = 'custom_weights_negclass_best_saved.h5'

# load trained model
model = load_model(file_path)

# load video
video_file = "IMG-6015.MOV"
image_gen = video_frame_generator(video_file)

image = image_gen.__next__()
image = np.ascontiguousarray(np.rot90(image, 3))
h, w, d = image.shape

out_path = "demo_video.mp4"
fps = 20
video_out = mp4_video_writer(out_path, (w, h), fps)

while image is not None:

    # rotate image 90 degrees
    image = np.ascontiguousarray(np.rot90(image, 3))

    # MSER detection approach
    # the following stackoverflow post was used as a reference for the MSER region extraction and bounding box code below:
    # https://stackoverflow.com/questions/47595684/extract-mser-detected-areas-python-opencv
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_max_variation = 0.55)
    mser.setMaxArea(100000)
    mser.setMinArea(1000)
    regions, _ = mser.detectRegions(gray_image)
    convexhulls = []
    for reg in regions:
        convexhulls.append(cv2.convexHull(reg.reshape(-1, 1, 2)))
    #cv2.polylines(image, convexhulls, 1, (0, 255, 0))
    boxes = []
    scores = []
    digit_preds = []
    if len(convexhulls) != 0:
        for i, cont in enumerate(convexhulls):
            x, y, w, h = cv2.boundingRect(cont)
            if w < 0.25*h:
                continue
            if h < 0.25*w:
                continue
            if w > h:
                y = y + (h/2) - (w/2)
                h = w
            else:
                x = x + (w/2) - (h/2)
                w = h
            if w*h < 1000:
                continue
            #cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 4)
            perc = 0.2
            xe1 = int(x-(perc*w))
            xe2 = int(x+w+(perc*w))
            ye1 = int(y-(perc*h))
            ye2 = int(y+h+(perc*h))
            window = image[ye1:ye2, xe1:xe2]
            try:
                window = cv2.resize(window, (32,32))
                sample = np.expand_dims(window, axis=0)
                sample = preprocess_input(sample, mode='tf')
                prediction = model.predict(sample)
                if np.max(prediction[:10]) > 0.95:
                    #cv2.rectangle(image, (xe1, ye1), (xe2, ye2), (0, 0, 255), 4)
                    #cv2.putText(image, str(np.argmax(prediction[:10])), org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255,0,0), thickness=4, lineType=cv2.LINE_AA)
                    boxes.append([ye1, xe1, ye2, xe2])
                    scores.append(np.max(prediction[:10]))
                    digit_preds.append(np.argmax(prediction[:10]))
            except:
                continue

    # non-max suppression
    num_boxes = 10
    iou_thresh = 0.5
    if len(boxes) != 0:
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=num_boxes, iou_threshold=iou_thresh)
        #sess = tf.compat.v1.Session()
        sess = tf.Session()
        with sess.as_default():
            nms_indices = nms_indices.eval()
        for ind in nms_indices:
            flag = 1
            y1, x1, y2, x2 = boxes[ind]
            # check if box is fully contained by other box
            for i in range(len(nms_indices)):
                if nms_indices[i] == ind:
                    continue
                else:
                    cy1, cx1, cy2, cx2 = boxes[nms_indices[i]]
                    if x1 > cx1 and x2 < cx2 and y1 > cy1 and y2 < cy2:
                        flag = 0
            if flag:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(image, str(digit_preds[ind]), org=(x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255,0,0), thickness=4, lineType=cv2.LINE_AA)


    video_out.write(image)

    image = image_gen.__next__()

video_out.release()
