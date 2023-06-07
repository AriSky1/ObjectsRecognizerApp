from dash import Dash, dcc, html, Input, Output, State
from flask import Flask, render_template, Response
import dash_bootstrap_components as dbc
from ultralytics import YOLO
from gen_frames import gen_frames_yolo






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
                        html.Button('Start', id='start_btn', n_clicks=0, className='btn btn-success'),
                        html.Div(id='container-stream')
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
    Input('start_btn', 'n_clicks')

)


def load_stream(n_clicks):
    if n_clicks > 0:
        # Your code to start the object recognition stream goes here
        # Replace the following line with your implementation
        # return html.Div("Object recognition stream has started.")
        return html.Div([

            html.Div([html.Img(id='stream', src="/stream")])
        ])



    else:
        return html.Div()






@server.route('/stream')
def stream():
    # url = 'https://www.youtube.com/watch?v=IBFCV4zhMGc' #shibuya static
    url = 0
    model = YOLO('yolov8n.pt')
    return Response(gen_frames_yolo(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run_server(debug=True)