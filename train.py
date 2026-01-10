from preprocess import load_data, generator
from model import nvidia_model

BATCH_SIZE = 32
EPOCHS = 10

train_samples, validation_samples = load_data()

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = nvidia_model()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_samples) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) // BATCH_SIZE,
    epochs=EPOCHS
)

model.save('model.h5')