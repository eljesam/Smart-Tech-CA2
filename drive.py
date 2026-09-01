import base64
from io import BytesIO
from pathlib import Path

import eventlet
import eventlet.wsgi
import numpy as np
import socketio

from flask import Flask
from PIL import Image
from keras.models import load_model

from src.preprocessing import preprocess_image


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "self_driving_model_run2.keras"
)

TARGET_SPEED = 8.0
THROTTLE_VALUE = 0.15
STEERING_GAIN = 1.0

sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*"
)

app = Flask(__name__)


print("\n========================================")
print("LOADING RUN 2 MODEL")
print("========================================")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Run 2 model not found:\n{MODEL_PATH}"
    )

model = load_model(
    MODEL_PATH,
    compile=False
)

print("\nRun 2 model loaded successfully.")
print(MODEL_PATH)


def send_control(
    sid,
    steering_angle,
    throttle
):

    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        },
        room=sid
    )


@sio.on("connect")
def connect(sid, environ):

    print("\n========================================")
    print("SIMULATOR CONNECTED")
    print("========================================")

    send_control(
        sid,
        0.0,
        0.0
    )


@sio.on("telemetry")
def telemetry(sid, data):

    if not data:

        sio.emit(
            "manual",
            data={},
            room=sid
        )

        return

    speed = float(
        data["speed"]
    )

    image_data = base64.b64decode(
        data["image"]
    )

    image = Image.open(
        BytesIO(image_data)
    )

    image = np.asarray(
        image
    )

    image = preprocess_image(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    raw_steering = float(
        prediction[0][0]
    )

    steering_angle = (
        raw_steering
        * STEERING_GAIN
    )

    steering_angle = np.clip(
        steering_angle,
        -1.0,
        1.0
    )

    if speed < TARGET_SPEED:
        throttle = THROTTLE_VALUE
    else:
        throttle = 0.0

    print(
        f"Speed: {speed:6.2f} | "
        f"Raw steering: {raw_steering:7.4f} | "
        f"Final steering: {steering_angle:7.4f} | "
        f"Throttle: {throttle:.2f}"
    )

    send_control(
        sid,
        steering_angle,
        throttle
    )


if __name__ == "__main__":

    wrapped_app = socketio.WSGIApp(
        sio,
        app
    )

    print("\n========================================")
    print("RUN 2 AUTONOMOUS DRIVING")
    print("========================================")

    print(f"Model: {MODEL_PATH.name}")
    print(f"Target speed: {TARGET_SPEED}")
    print(f"Throttle: {THROTTLE_VALUE}")
    print(f"Steering gain: {STEERING_GAIN}")

    print("\nListening on port 4567...")
    print("Start the simulator in Autonomous Mode.\n")

    eventlet.wsgi.server(
        eventlet.listen(
            ("", 4567)
        ),
        wrapped_app
    )