from dash import Dash, dcc, html, Input, Output, State
from flask import Flask, render_template, Response
import dash_bootstrap_components as dbc
from ultralytics import YOLO
from gen_frames import gen_frames_yolo
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import matplotlib.pyplot as plt
import random
import colorsys

model = YOLO("yolov8n.pt")

# Generate unique colors for each label
def generate_label_colors(num_labels):
    hsv_colors = [(i / num_labels, 1, 1) for i in range(num_labels)]
    rgb_colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv_colors))
    bgr_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]
    return bgr_colors

def gen_frames_yolo():


    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    label_colors = generate_label_colors(len(model.names))

    while cap.isOpened():
        ret, frame = cap.read()

        # model.predict(source="0", show=True)
        results = model(frame)  # Perform object detection on the frame
        # print(results)
        for r in results:

            annotator = Annotator(frame)


            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                label = model.names[int(c)]
                color = label_colors[int(c)]


                annotator.box_label(b, model.names[int(c)],color=color)

        frame = annotator.result()

        frame = cv2.imencode('.jpg', frame)[1].tobytes()





        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # key = cv2.waitKey(20)
        # if key == 27:
        #     break



external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@600&display=swap',
     dbc.themes.CYBORG
]

style_title={'color': 'grey','fontSize': 30,'textAlign': 'center', 'letter-spacing':'2px', 'padding-left': '20px','padding-top': '20px'}
style_text={'color': 'grey','fontSize': 18,'textAlign': 'center','font_family': 'Segoe UI', 'padding-bottom':'20px','padding-left': '20px'}
style_btn = {'color': 'black','font-weight': 'bold', 'width':'200px', 'height':'100px',}

server = Flask(__name__)
app = Dash(__name__, server=server,external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.H1(children='Objects Recognizer', style=style_title),
                        html.Div(
                            children='Recognize multiple objects from your live web cam stream.',
                            style=style_text
                        ),
                        html.Button('Start', id='start_btn', n_clicks=0, className='btn btn-success',style={"margin-bottom": "30px"}),

                        html.Div(id='container-stream'),

                        html.Img(id='stream', style={'width': '50px', 'height': '50px', 'display': 'none'})
                    ],
                    className='my-5 text-center'
                )
            ],
            fluid=True
        )
    ],
    style={"display": "flex", "flex-direction": "column", "align-items": "center"}
)


@app.callback(
    Output('container-stream', 'children'),
    Input('start_btn', 'n_clicks'),


)


def load_stream(n_clicks):
    if n_clicks > 0:
        return html.Div([

            html.Img(id='stream', src="/stream")
        ])
    else:
        return html.Div()












@server.route('/stream')
def stream():
    return Response(gen_frames_yolo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run_server(debug=True)