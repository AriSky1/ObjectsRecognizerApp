from dash import Dash, dcc, html, Input, Output, State
from flask import Flask, render_template, Response
import dash_bootstrap_components as dbc
from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

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
        results = model(frame)  # Perform object detection on the frame
        for r in results:
            annotator = Annotator(frame)
            li=[]
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                label = model.names[int(c)] #person, couch, chain in loop
                color = label_colors[int(c)]
                li.append(label)
                annotator.box_label(b, model.names[int(c)],color=color)

            d = {i: li.count(i) for i in li}

            d1 = dict(list(d.items())[:3])
            # new_dic2 = list(d.items())[4:]
            # new_dic1 = dict(list(d.items())[:4])
            d2 = dict(list(d.items())[3:])


            d1 = str(d1)
            d2 = str(d2)

            d1 = d1.replace('}', ' ')
            d1 = d1.replace('{', ' ')
            d1 = d1.replace("'", ' ')
            d1 = d1.replace(",", '')


            d2 = d2.replace('}', ' ')
            d2 = d2.replace('{', ' ')
            d2 = d2.replace("'", ' ')
            d2 = d2.replace(",", '')


            #
            cv2.putText(frame, str(d1), (10,30), cv2.FONT_HERSHEY_DUPLEX,
                                    0.9, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(d2), (30,70), cv2.FONT_HERSHEY_DUPLEX,
                                    0.9, (0, 0, 0), 1, cv2.LINE_AA)

        frame = annotator.result()
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame  + b'\r\n')





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


style_title={'color': 'grey','fontSize': 30,'textAlign': 'center', 'letter-spacing':'2px', 'padding-left': '20px','padding-top': '20px'}
style_text={'color': 'grey','fontSize': 18,'textAlign': 'center','font_family': 'Segoe UI', 'padding-bottom':'20px','padding-left': '20px'}
style_btn = {'color': 'grey','font-weight': 'bold', 'width':'100px', 'height':'50px',}

server = Flask(__name__)
app = Dash(__name__, server=server,external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.H1(children='Objects Counter', style=style_title),
                        html.Div(
                            children='Count multiple objects from your web cam.',
                            style=style_text
                        ),
                        html.Div(
                            html.Button('Start', id='start_btn', n_clicks=0, className='btn btn-success', style=style_btn),
                            style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '30px',"background-color": "black"}
                        ),

                        html.Div(id='container-stream', style={'margin':'0px'}),
                        html.Div(id='dict-placeholder'),




                        html.Img(id='stream', style={'width': '50px', 'height': '50px', 'display': 'none'})
                    ],
                    className='my-5 text-center'
                )
            ],
            fluid=True,
            style={"height": "100vh", "display": "flex", "flex-direction": "column", "align-items": "center", "background-color": "black", "margin": '0px'}
        )
    ],
    style={"display": "flex", "flex-direction": "column", "align-items": "center", "background-color": "black", "margin": '0px'}
)





@app.callback(
    Output('container-stream', 'children'),
    Input('start_btn', 'n_clicks'),



)

def load_stream(n_clicks):
    if n_clicks > 0:
        return html.Div([

            html.Img(id='stream', src="/stream"),
        ])
    else:
        return html.Div()









# @app.callback(
#     Output('dict-placeholder', 'children'),
#     Input('start_btn', 'n_clicks'),
#
# )
#
# def get_d(n_clicks):
#
#
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         results = model(frame)  # Perform object detection on the frame
#         for r in results:
#             li = []
#             boxes = r.boxes
#             for box in boxes:
#                 b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
#                 c = box.cls
#                 label = model.names[int(c)]  # person, couch, chain in loop
#                 li.append(label)
#             d = {i: li.count(i) for i in li}
#     return html.Div(d)
#






@server.route('/stream')
def stream():
    return Response(gen_frames_yolo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run_server(debug=True)