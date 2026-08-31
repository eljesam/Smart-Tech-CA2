import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import load_image, preprocess_image
from augmentation import augment_sample
from model import build_model

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
CSV_PATH = DATA_DIR / "driving_log_balanced.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "self_driving_model.keras"

df = pd.read_csv(CSV_PATH)

print("Total balanced samples:", len(df))

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))