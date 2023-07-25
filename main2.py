
from dash import Dash, dcc, html, Input, Output, State
from flask import Flask, render_template, Response
import dash_bootstrap_components as dbc
from ultralytics import YOLO
# from gen_frames import gen_frames_yolo
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import matplotlib.pyplot as plt
import random
import colorsys



# Generate unique colors for each label
def generate_label_colors(num_labels):
    hsv_colors = [(i / num_labels, 1, 1) for i in range(num_labels)]
    rgb_colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv_colors))
    bgr_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]
    return bgr_colors


model = YOLO("yolov8n.pt")



cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

label_colors = generate_label_colors(len(model.names))

while cap.isOpened():
    ret, frame = cap.read()

    # model.predict(source="0", show=True)
    results = model(frame)  # Perform object detection on the frame
    # print(results)
    for r in results:

        annotator = Annotator(frame)

        li = []
        coord = [500, 500, 500, 500]
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            label = model.names[int(c)]  # person, couch, chain in loop
            color = label_colors[int(c)]

            li.append(label)
            my_dict = {i: li.count(i) for i in li}

            print(my_dict)
            print(my_dict.keys())
            print(my_dict.values())

            annotator.box_label(b, model.names[int(c)], color=color)
            print(b)

            annotator.box_label(b, model.names[int(c)], color=color)



    frame = annotator.result()

    frame = cv2.imencode('.jpg', frame)[1].tobytes()