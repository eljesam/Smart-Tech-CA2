import base64

from io import BytesIO
from pathlib import Path

import eventlet
import eventlet.wsgi
import numpy as np
import socketio

from flask import Flask
from keras.models import load_model
from PIL import Image

from src.preprocessing import preprocess_image

#Socket.IO/Flask setup
sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*"
)

flask_app = Flask(__name__)

#Project Paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "self_driving_model_run2.keras"
)

#Driving settings
TARGET_SPEED = 8.0
THROTTLE_VALUE = 0.15

STEERING_GAIN = 1.0
STEERING_BIAS = 0.0

# Control Function
def send_control(sid, steering_angle, throttle):
   
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        },
        to=sid
    )

# Simulator connection
@sio.event
def connect(sid, environ):
    
    print("Simulator connected.")

    send_control(
        sid,
        steering_angle=0.0,
        throttle=0.0
    )

@sio.on("telemetry")
def telemetry(sid, data):
  
    if not data:
        sio.emit(
            "manual",
            data={},
            to=sid
        )
        return

    # Current simulator speed
    speed = float(data["speed"])

    # Decode the center camera image
    image_data = base64.b64decode(
        data["image"]
    )

    image = Image.open(
        BytesIO(image_data)
    )

    image = np.asarray(image)

    
    image = preprocess_image(image)

  
    image = np.expand_dims(
        image,
        axis=0
    ).astype(np.float32)

    # Predict steering angle
    prediction = model.predict(
        image,
        verbose=0
    )

    steering_angle = float(
        prediction[0][0]
    )

    
    steering_angle *= STEERING_GAIN
   

    steering_angle = np.clip(
        steering_angle,
        -1.0,
        1.0
    )

    # Simple speed controller
    if speed < TARGET_SPEED:
        throttle = THROTTLE_VALUE
    else:
        throttle = 0.0

    print(
        f"Speed: {speed:6.2f} | "
        f"Steering: {steering_angle:7.4f} | "
        f"Throttle: {throttle:.2f}"
    )

    send_control(
        sid,
        steering_angle,
        throttle
    )

if __name__ == "__main__":

    print("Loading model:")
    print(MODEL_PATH)

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print("Waiting for simulator...")
    print("Port: 4567")

    app = socketio.WSGIApp(
        sio,
        flask_app
    )

    eventlet.wsgi.server(
        eventlet.listen(
            ("", 4567)
        ),
        app
    )