import argparse
import base64
from io import BytesIO

import numpy as np
import cv2

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from PIL import Image

import keras
keras.config.enable_unsafe_deserialization()

from tensorflow.keras.models import load_model


def preprocess_image(img_bgr):
    img_bgr = img_bgr[60:135, :, :]  # Crop the image
    # Convert to YUV color space
    img_bgr = cv2.resize(img_bgr, (200, 66))
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    return img_yuv

sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
model = None

@sio.on('connect')
def connect(sid, environ):
    print("Client connected: ", sid)
    send_control(0.0, 0.0)
    
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )
    
@sio.on('telemetry')
def telemetry(sid, data):
    global model
    
    if data is None:
        return
    
    img_str = data["image"]
    img = Image.open(BytesIO(base64.b64decode(img_str)))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img = preprocess_image(img)
    
    x = np.expand_dims(img, axis=0).astype(np.float32)
    
    steering_angle = float(model.predict(x, verbose=0)[0])
    
    if abs(steering_angle) < 0.1:
        throttle = 0.3
    elif abs(steering_angle) < 0.3:
        throttle = 0.2
    else:
        throttle = 0.15
        
    send_control(steering_angle, throttle)

def main():
    global model
    
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    
    print("Loading model from ", args.model)
    model = load_model(args.model, compile=False, safe_mode=False)
    
    app_wrapped = socketio.Middleware(sio, app)
    
    print("Starting server on port 4567...")

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app_wrapped)

if __name__ == '__main__':
    main()
        
