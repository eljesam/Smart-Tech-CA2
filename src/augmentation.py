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

#Brightness Augmentation
def random_brightness(image):
    

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    brightness_factor = np.random.uniform(
        0.6,
        1.4
    )

    hsv = hsv.astype(np.float32)

    hsv[:, :, 2] *= brightness_factor

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2],
        0,
        255
    )

    hsv = hsv.astype(np.uint8)

    image = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    return image

#Generating augmented training sample
def augment_sample(row):
  

    image, steering = select_camera(row)

    image, steering = random_flip(
        image,
        steering
    )

    image = random_brightness(
        image
    )

    return image, steering

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    CSV_PATH = (
        BASE_DIR
        / "data"
        / "raw"
        / "driving_log_balanced.csv"
    )

    df = pd.read_csv(
        CSV_PATH
    )

    row = df.iloc[100]

    image, steering = augment_sample(
        row
    )

    print(
        "Original steering:",
        row["steering"]
    )

    print(
        "Augmented steering:",
        steering
    )

    print(
        "Image shape:",
        image.shape
    )

    plt.imshow(image)

    plt.title(
        f"Augmented Image - Steering: {steering:.3f}"
    )

    plt.axis("off")
    plt.show()