import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import load_image, preprocess_image
from augmentation import augment_sample
from model import build_model


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
CSV_PATH = DATA_DIR / "driving_log_balanced.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "self_driving_model.keras"



# Training settings


BATCH_SIZE = 32
EPOCHS = 20



# Training generator


def training_generator(dataframe, batch_size=32):
  
    while True:

        batch_indices = np.random.choice(
            len(dataframe),
            batch_size,
            replace=True
        )

        images = []
        steering_angles = []

        for index in batch_indices:

            row = dataframe.iloc[index]

            # Random camera selection
         
            image, steering = augment_sample(row)

            # Crop, resize, convert RGB -> YUV
            image = preprocess_image(image)

            images.append(image)
            steering_angles.append(steering)

        yield (
            np.array(images, dtype=np.float32),
            np.array(steering_angles, dtype=np.float32)
        )


# Validation generator


def validation_generator(dataframe, batch_size=32):
 
    current_index = 0

    while True:

        images = []
        steering_angles = []

        for _ in range(batch_size):

            if current_index >= len(dataframe):
                current_index = 0

            row = dataframe.iloc[current_index]

            image = load_image(
                row["center"]
            )

            steering = float(
                row["steering"]
            )

            image = preprocess_image(image)

            images.append(image)
            steering_angles.append(steering)

            current_index += 1

        yield (
            np.array(images, dtype=np.float32),
            np.array(steering_angles, dtype=np.float32)
        )



# Load dataset


df = pd.read_csv(CSV_PATH)

print("Total balanced samples:", len(df))


# Train / validation split


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))



# Calculate training steps


STEPS_PER_EPOCH = max(
    1,
    len(train_df) // BATCH_SIZE
)

VALIDATION_STEPS = max(
    1,
    len(val_df) // BATCH_SIZE
)

print("Steps per epoch:", STEPS_PER_EPOCH)
print("Validation steps:", VALIDATION_STEPS)



# Create generators


train_gen = training_generator(
    train_df,
    batch_size=BATCH_SIZE
)

val_gen = validation_generator(
    val_df,
    batch_size=BATCH_SIZE
)



# Build model


print("\nBuilding model...\n")

model = build_model()

model.summary()



# Callbacks


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True
)



# Train model


print("\nStarting training...\n")

history = model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_gen,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)



# Save final model

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)



# Plot training and validation loss


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plot_path = MODEL_DIR / "training_loss.png"

plt.savefig(
    plot_path,
    bbox_inches="tight"
)

print("\nTraining graph saved to:")
print(plot_path)

plt.show()



# Print training summary


best_val_loss = min(
    history.history["val_loss"]
)

best_epoch = (
    history.history["val_loss"].index(
        best_val_loss
    ) + 1
)

print("\nTraining Summary")
print("----------------")
print("Epochs completed:", len(history.history["loss"]))
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)