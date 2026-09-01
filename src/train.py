import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import load_image, preprocess_image
from augmentation import augment_sample
from model import build_model

from keras.models import load_model
from keras.optimizers import Adam

# Project paths


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

CSV_PATH = DATA_DIR / "driving_log_run6_finetune.csv"

BASE_MODEL_PATH = (
    MODEL_DIR / "self_driving_model_run2.keras"
)

MODEL_PATH = (
    MODEL_DIR / "self_driving_model_run6.keras"
)

PLOT_PATH = (
    MODEL_DIR / "training_loss_run6.png"
)


MODEL_DIR.mkdir(parents=True, exist_ok=True)



# Training settings

BATCH_SIZE = 32
EPOCHS = 8
VALIDATION_SPLIT = 0.20

RANDOM_STATE = 42

# Training generator


def training_generator(dataframe, batch_size=32):

    while True:

        images = []
        steering_angles = []

        for _ in range(batch_size):

            # Randomly select a row
            index = np.random.randint(
                0,
                len(dataframe)
            )

            row = dataframe.iloc[index]

            # Apply augmentation
            image, steering = augment_sample(row)

            # Apply image preprocessing
            image = preprocess_image(image)

            images.append(image)
            steering_angles.append(steering)

        X_batch = np.array(
            images,
            dtype=np.float32
        )

        y_batch = np.array(
            steering_angles,
            dtype=np.float32
        )

        yield X_batch, y_batch


# Validation generator


def validation_generator(dataframe, batch_size=32):
 
 current_index = 0

 while True:

        images = []
        steering_angles = []

        for _ in range(batch_size):

            # Restart from beginning when end is reached
            if current_index >= len(dataframe):
                current_index = 0

            row = dataframe.iloc[current_index]

            # Load center camera image
            image = load_image(
                row["center"]
            )

            # Apply preprocessing only
            image = preprocess_image(
                image
            )

            steering = float(
                row["steering"]
            )

            images.append(image)
            steering_angles.append(steering)

            current_index += 1

        X_batch = np.array(
            images,
            dtype=np.float32
        )

        y_batch = np.array(
            steering_angles,
            dtype=np.float32
        )

        yield X_batch, y_batch



# Load dataset
print("\n========================================")
print("RUN 6 TRAINING")
print("========================================")

print("\nLoading dataset:")
print(CSV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Run 5 dataset was not found:\n{CSV_PATH}"
    )


df = pd.read_csv(
    CSV_PATH
)


print("\nDataset loaded successfully.")
print("Total samples:", len(df))

print("\nDataset columns:")
print(df.columns.tolist())


# Train / validation split

train_df, validation_df = train_test_split(
    df,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
    shuffle=True
)


# Reset indexes
train_df = train_df.reset_index(
    drop=True
)

validation_df = validation_df.reset_index(
    drop=True
)
print("\n========================================")
print("DATA SPLIT")
print("========================================")

print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(validation_df)
)

train_generator = training_generator(
    train_df,
    BATCH_SIZE
)

val_generator = validation_generator(
    validation_df,
    BATCH_SIZE
)

# Calculate training steps


steps_per_epoch = max(
    1,
    len(train_df) // BATCH_SIZE
)

validation_steps = max(
    1,
    len(validation_df) // BATCH_SIZE
)


print("\nSteps per epoch:",
      steps_per_epoch)

print(
    "Validation steps:",
    validation_steps
)


# Build model


print("\nBuilding model...\n")

model = load_model(
    BASE_MODEL_PATH,
    compile=False
)

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="mse"
)



# Callbacks


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [
    early_stopping,
    model_checkpoint
]




# Train model


print("\nStarting training...\n")


history = model.fit(

    train_generator,

    steps_per_epoch=steps_per_epoch,

    validation_data=val_generator,

    validation_steps=validation_steps,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# Save final model

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

# Training statistics

training_loss = history.history[
    "loss"
]

validation_loss = history.history[
    "val_loss"
]


best_epoch = (
    np.argmin(validation_loss) + 1
)

best_validation_loss = np.min(
    validation_loss
)
print("\n========================================")
print("RUN 5 RESULTS")
print("========================================")

print(
    f"Epochs completed: "
    f"{len(training_loss)}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
)

# Plot training and validation loss


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    training_loss,
    label="Training Loss"
)

plt.plot(
    validation_loss,
    label="Validation Loss"
)

plt.title(
    "Run 5 - Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    PLOT_PATH
)


print("\nTraining graph saved:")
print(PLOT_PATH)


plt.show()



# Print training summary

print("\n========================================")
print("RUN 5 TRAINING COMPLETE")
print("========================================")

print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(validation_df)}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
)

print(
    f"Model: {MODEL_PATH.name}"
)

print(
    f"Loss graph: {PLOT_PATH.name}"
)