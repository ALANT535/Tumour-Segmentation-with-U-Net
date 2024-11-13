from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([

    Conv2D(32, (4, 4), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(64, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(128, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    Conv2D(128, (4, 4), activation="relu"),
    Flatten(),

    Dense(512, activation="relu"),
    Dropout(0.5, seed=42),
    
    Dense(1, activation="sigmoid")
])

model.load_weights('SAVED_MODEL.h5')

model.save('SAVED_MODEL.h5')