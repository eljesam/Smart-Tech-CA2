import os
import pandas as pd
import cv2

CSV = "data/driving_log_relative.csv"   # change if you're using a different one
IMG_DIR = "data/IMG"

df = pd.read_csv(CSV)

print("IMG folder exists:", os.path.isdir(IMG_DIR))
print("IMG files count:", len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else 0)

def to_full(p):
    p = str(p).strip().replace("\\", "/")
    # force it to be IMG/<filename>
    filename = p.split("/")[-1]
    return os.path.join("data", "IMG", filename)

for i in range(10):
    full = to_full(df.iloc[i]["center"])
    img = cv2.imread(full)
    print(i, full, "OK" if img is not None else "MISSING")
