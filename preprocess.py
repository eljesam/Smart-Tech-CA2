import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn

DATA_PATH = 'data/'
CORRECTION = 0.2  # Steering correction for left and right images

def load_data():
    data = pd.read_csv(DATA_PATH + 'balanced_driving_log.csv') 
    return train_test_split(data, test_size=0.2, random_state=42)

def preprocess_image(image):
    image = image[60:-25, :, :]  # Crop
    image = cv2.resize(image, (200, 66))  # Resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Color space conversion
    return image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            
            for _, row in batch_samples.iterrows():
                steering = row['steering']
                paths = [row['center'], row['left'], row['right']]
                corrections = [0, CORRECTION, -CORRECTION]
                for path, corr in zip(paths, corrections):
                    filename = path.split('/')[-1]
                    image = cv2.imread(DATA_PATH + 'IMG/' + filename)
                    image = preprocess_image(image)
                    angle = steering + corr
                    
                    # Data augmentation: flip image
                    images.append(image)
                    angles.append(angle)
                
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)
                    
                yield sklearn.utils.shuffle(np.array(images), np.array(angles))