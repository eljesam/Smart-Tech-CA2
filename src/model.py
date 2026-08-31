import tensorflow as tf

from keras.models import Sequential
from keras.layers import (
    Input,
    Rescaling,
    Conv2D,
    Flatten,
    Dense,
    Dropout
)
from keras.optimizers import Adam

def build_model():
   

    model = Sequential([
        Input(shape=(66, 200, 3)),

        # Normalize pixel values from 0-255 to approximately -1 to 1
        Rescaling(
            scale=1.0 / 127.5,
            offset=-1
        ),

        # Convolutional layers
        Conv2D(
            24,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation="elu"
        ),

        Conv2D(
            36,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation="elu"
        ),

        Conv2D(
            48,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation="elu"
        ),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="elu"
        ),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="elu"
        ),

        Flatten(),

        # Fully connected layers
        Dense(
            100,
            activation="elu"
        ),

        Dropout(0.5),

        Dense(
            50,
            activation="elu"
        ),

        Dense(
            10,
            activation="elu"
        ),

        # Steering angle output
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss="mse"
    )

    return model

if __name__ == "__main__":
    print("Building model...")

    model = build_model()

    print("Model built successfully.\n")

    model.summary()