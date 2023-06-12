import io
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import imageio
import base64

# Cargar modelo
model = YOLO('best.pt')

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("Peticion recibida")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        binary_image = post_data.split(b'\r\n\r\n')[1]

        # Decodificar imagen a PIL
        pil_image = Image.open(io.BytesIO(binary_image))

        # Generar prediccion
        results = model(source=pil_image)
        res_plot = results[0].plot()
        img = Image.fromarray(res_plot)
        imageio.imwrite('output.jpg', img)

        # Leer la imagen desde el archivo
        with open('output.jpg', 'rb') as file:
            image_data = file.read()
            image_data = base64.b64encode(image_data)

        # Generar respuesta a la petición HTTP
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", "image/jpeg")
        self.send_header("Content-length", len(image_data))
        self.end_headers()

        # Enviar la imagen
        self.wfile.write(image_data)

print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()

