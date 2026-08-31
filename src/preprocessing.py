import cv2
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
IMG_DIR = DATA_DIR / "IMG"

#creating image loader
def load_image(image_path):
    
    
    filename = Path(str(image_path).strip()).name
    full_path = IMG_DIR / filename

    image = cv2.imread(str(full_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {full_path}")

    # OpenCV loads BGR, convert it to RGB for processing/display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

#Crop
def crop_image(image):
   

    return image[60:135, :, :]

#Resize
def resize_image(image):
   

    return cv2.resize(
        image,
        (200, 66)
    )

#Colour conversion
def convert_to_yuv(image):
    

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2YUV
    )

#Preprocessing Image
def preprocess_image(image):
    

    image = crop_image(image)

    image = resize_image(image)

    image = convert_to_yuv(image)

    return image

if __name__ == "__main__":

    import pandas as pd

    CSV_PATH = DATA_DIR / "driving_log_balanced.csv"

    df = pd.read_csv(CSV_PATH)

    image = load_image(
        df.iloc[0]["center"]
    )

    print("Original shape:", image.shape)

    processed = preprocess_image(image)

    print("Processed shape:", processed.shape)