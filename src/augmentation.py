import cv2
import numpy as np

from preprocessing import load_image

STEERING_CORRECTION = 0.2

#Camera selection
def select_camera(row):
   

    steering = float(row["steering"])

    camera = np.random.choice(
        ["center", "left", "right"]
    )

    if camera == "left":
        steering += STEERING_CORRECTION

    elif camera == "right":
        steering -= STEERING_CORRECTION

    image = load_image(row[camera])

    return image, steering

#Horizontal flipping

def random_flip(image, steering):
 
    if np.random.rand() < 0.5:

        image = cv2.flip(
            image,
            1
        )

        steering = -steering

    return image, steering