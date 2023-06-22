import cv2
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Button('Start', id='start-button'),
    html.Img(id='video-stream', style={'display': 'none'}),
    dcc.Interval(id='interval', interval=1000, n_intervals=0)
])


@app.callback(
    Output('video-stream', 'src'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def start_video_stream(n_clicks):
    if n_clicks % 2 != 0:
        cap = cv2.VideoCapture(0)
        _, frame = cap.read()
        height, width, _ = frame.shape
        fps = 30

        model = YOLO('yolov8n.pt')
        annotator = Annotator(frame)

        def gen_frames():
            while True:
                _, frame = cap.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = model.predict(img)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])

                annotated_frame = annotator.result()
                _, encoded_frame = cv2.imencode('.jpg', annotated_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')

        return '/stream'


@app.callback(
    Output('video-stream', 'style'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_video_stream(n_clicks):
    if n_clicks % 2 != 0:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('interval', 'disabled'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def disable_interval(n_clicks):
    if n_clicks % 2 != 0:
        return False
    else:
        return True


@app.callback(
    Output('start-button', 'children'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_button_text(n_clicks):
    if n_clicks % 2 != 0:
        return 'Stop'
    else:
        return 'Start'


@app.callback(
    Output('video-stream', 'src'),
    Output('interval', 'interval'),
    Input('interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_video_stream(n_intervals):
    return '/stream', 1000


if __name__ == '__main__':
    app.run_server(debug=True)