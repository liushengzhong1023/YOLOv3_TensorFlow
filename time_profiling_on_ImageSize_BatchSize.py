# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import os

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(
    description="YOLO-V3 inference time efficiency profiling w.r.t image sizes and batch sizes.")

parser.add_argument("--input_image", type=str,
                    default="./data/demo_data/messi.jpg",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str,
                    default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", type=int,
                    default=416,
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--batch_size", type=int,
                    default=1,
                    help="Specify the number of images fed in each batch.")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'),
                    default=False,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str,
                    default="./data/my_data/COCO/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str,
                    default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

# ----------------------------------------------------------------------------------------------------------------------

args = parser.parse_args()
args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)
args.new_size = [args.new_size, args.new_size]

color_table = get_color_table(args.num_class)

# read and resize image
img_ori = cv2.imread(args.input_image)
if args.letterbox_resize:
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
else:
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)

# normalization --> [0, 1]
img = img[np.newaxis, :] / 255.
img = np.tile(img, [args.batch_size, 1, 1, 1])

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [None, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                    nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    # warm up run
    for _ in range(5):
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # time profiling
    start = time.time()

    for _ in range(100):
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    end = time.time()
    print("------------------------------------------------------------------------")
    avg_inference_time = (end-start) / float(100)
    avg_throughput = 1 / avg_inference_time * args.batch_size
    print("Average execution time: %f s" % avg_inference_time)
    print("Average throughput: %f" % avg_throughput)
