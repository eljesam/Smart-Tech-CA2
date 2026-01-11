import tensorflow as tf
Sequential = tf.keras.Sequential
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Rescaling = tf.keras.layers.Rescaling
Adam = tf.keras.optimizers.Adam

def nvidia_model():
    model = Sequential()

    # Normalization layer
    model.add(Rescaling(1./255, offset=-0.5, input_shape=(66,200,3)))
    
    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model