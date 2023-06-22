
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
from dash import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash import ctx
from dash.dependencies import Input, Output, State
# import pafy
import cv2
from flask import Flask, render_template, Response
from datetime import datetime
import pytz
import time
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import supervision as sv

def gen_frames_yolo():


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    model = YOLO("yolov8n.pt")


    while cap.isOpened():
        ret, frame = cap.read()

        results = model(frame)  # Perform object detection on the frame

        # Draw bounding boxes on the frame
        for label, confidence, bbox in results.xyxy[0]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


