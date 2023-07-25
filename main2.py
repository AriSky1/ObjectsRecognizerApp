import dash
from dash import html, dcc, Input, Output, State
from dash.dependencies import Input, Output
import cv2
from ultralytics import YOLO

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Chargement du modèle YOLOv8
model = YOLO("yolov8n.pt")

# Callback pour démarrer la webcam et effectuer la détection d'objets
@app.callback(
    Output('detected-objects', 'children'),
    Input('start-btn', 'n_clicks'),
    State('webcam-video', 'children')
)
def start_webcam(n_clicks, video):
    if n_clicks is None:
        return ''

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)

    # Boucle pour capturer et traiter chaque image de la webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Effectuer la détection d'objets avec YOLOv8
        results = model(frame)

        # Liste des objets détectés
        detected_objects = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                detected_objects.append(label)

        # Convertir la liste en une chaîne de caractères
        detected_objects_str = ', '.join(detected_objects)

        # Afficher la liste des objets détectés dans le html.Div
        return detected_objects_str






# Mise en page de l'application Dash
app.layout = html.Div([
    html.H1("Détection d'objets avec YOLOv8"),
    html.Button('Start', id='start-btn', n_clicks=0),
    html.Div(id='webcam-video'),
    html.Div(id='detected-objects')
])

# Lancer l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)