# import required packages
import cv2
import argparse
import numpy as np
import os
from flask import Flask, request, url_for
from werkzeug.utils import secure_filename
import random

# handle command line arguments
ap = argparse.ArgumentParser()

ap.add_argument('--config', default='/var/www/FlaskApp/FlaskApp/yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('--weights', default='/var/www/FlaskApp/FlaskApp/yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('--classes', default='/var/www/FlaskApp/FlaskApp/yolov3.txt',
                help='path to text file containing class names')
args = ap.parse_args()

classes = None

Bounding_Boxes = []

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_duplicates_indeces(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def get_unique_values(lst):
    output = []
    for x in lst:
        if x not in output:
            output.append(x)
    return output


def split_classes(class_ids, boxes, confidences):
    all_ids = []
    all_boxes = []
    all_confidences = []
    unique_values = get_unique_values(class_ids)
    for value in unique_values:
        class_indeces = get_duplicates_indeces(class_ids, value)
        class_boxes = []
        class_confidences = []
        for index in class_indeces:
            class_boxes.append(boxes[index])
            class_confidences.append(confidences[index])
        all_boxes.append(class_boxes)
        all_confidences.append(class_confidences)
        all_ids.append(value)
    return all_ids, all_boxes, all_confidences


def process_frame(frame):
    image = cv2.imread(frame)
    # print(image[0][0])
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(args.weights, args.config)

    # image, scale, resize, subtruct mean, TRUE for BGR, no crop
    # [1, 3, 416, 416]
    # 1: batch number

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.5
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    classes_ids, classes_boxes, classes_confidences = split_classes(class_ids, boxes, confidences)
    for class_index in range(len(classes_ids)):
        class_indices = cv2.dnn.NMSBoxes(classes_boxes[class_index], classes_confidences[class_index],
                                         conf_threshold, nms_threshold)
        for box in classes_boxes[class_index]:
            for valid_box_index in class_indices:
                valid_box_index = valid_box_index[0]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                Bounding_Boxes.append([x, y, w, h])
    return Bounding_Boxes


app = Flask(__name__)
@app.route("/process_frame", methods=['GET', 'POST'])
def process():
    frame = request.files['frame']
    filename = str(random.randint(1,99999)) + frame.filename
    filePath = os.path.join("/var/www/FlaskApp/FlaskApp", filename)
    frame.save(filePath)
    boxes = process_frame(filePath)
    print('boxes are', boxes);
    print(boxes)
    return jsonify(boxes=boxes)

if __name__ == "__main__":
    app.run()
