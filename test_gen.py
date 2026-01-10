from preprocess import load_data, generator
import numpy as np

train_samples, _ = load_data()
gen = generator(train_samples, batch_size=32)

X, y = next(gen)

print("X type:", type(X), "shape:", X.shape, "dtype:", X.dtype)
print("y type:", type(y), "shape:", y.shape, "dtype:", y.dtype)

# sanity checks
assert len(X.shape) == 4 and X.shape[1:] == (66, 200, 3)
assert len(y.shape) == 1
print("Generator output looks correct")