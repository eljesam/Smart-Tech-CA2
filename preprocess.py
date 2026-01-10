import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn

DATA_PATH = "data/"
CORRECTION = 0.2  # Steering correction for left and right images

def load_data():
    data = pd.read_csv(DATA_PATH + 'driving_log_relative.csv') 
    return train_test_split(data, test_size=0.2, random_state=42)

def preprocess_image(image):
    image = image[60:-25, :, :]  # Crop
    image = cv2.resize(image, (200, 66))  # Resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Color space conversion
    return image

def read_image(rel_path):
    rel_path = str(rel_path).strip().replace('\\', '/')
    
    if "IMG/" in rel_path:
        rel_path = "IMG/" + rel_path.split("IMG/")[-1]

    full_path = os.path.join(DATA_PATH, rel_path)
    image = cv2.imread(full_path)
    return image, full_path

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            
            for _, row in batch_samples.iterrows():
                steering = float (row['steering'])
                
                paths = [row['center'], row['left'], row['right']]
                corrections = [0.0, CORRECTION, -CORRECTION]
                
                for path, corr in zip(paths, corrections):
                    image, full_path = read_image(path)
                    
                    if image is None:
                        print(f"Warning: Could not read image at {full_path}")
                        continue
                    
                    image = preprocess_image(image)
                    angle = steering + corr
                    # Data augmentation: flip image
                    images.append(image)
                    angles.append(angle)
                
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)
                 
                if len(images) == 0:
                # if nothing loaded in this batch, skip it
                 continue

            X = np.array(images, dtype=np.float32)
            y = np.array(angles, dtype=np.float32)

            yield (X, y)