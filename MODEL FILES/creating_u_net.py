import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt

# U-Net model function (architecture)
def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    
    # this is the downward part of the U-NET
    # Adding multiple conv blocks followed by max-pooling to capture features at different levels
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Bottleneck (Middle of the U-Net)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    
    # this is the upward part of the U-NET
    # Here we upsample and concatenate with the encoder layers to get better spatial details
    up5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    
    # Final output layer
    # The output layer has 1 filter (for binary segmentation) and uses sigmoid activation to give probability map
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    # Define the model and return the model to be used
    return model

# Load and preprocess data
def load_data(image_paths, mask_paths, image_size=(256, 256)):
    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        # resize the image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size)
        images.append(image)
        
        # resize the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size)
        masks.append(mask)
    
    # normalize them to [0,1] range
    images = np.array(images).astype('float32') / 255.0
    masks = np.array(masks).astype('float32') / 255.0
    
    # adding extra dimension
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)
    
    return images, masks

# infer using the model
def predict_segmentation(model, image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (256, 256))
    image_input = image_resized.astype('float32') / 255.0
    image_input = np.expand_dims(image_input, axis=(0, -1))
    
    # Make prediction
    prediction = model.predict(image_input)
    
    # Post-process the output
    prediction = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Binary threshold
    
    # Display the input image and the segmentation result side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized, cmap='gray')
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

# instantiate the model
unet = unet_model()
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# see structure of the model
unet.summary()

# Paths to your training images and masks
image_paths = os.listdir(r"images")
mask_paths = [[os.listdir(os.path.join("masks" , i))] for i in image_paths]

# loading data
train_images, train_masks = load_data(image_paths, mask_paths)

# Train the model
# We can now train the model on our images and masks data
unet.fit(train_images, train_masks, epochs=10, batch_size=8, validation_split=0.2)


# Test the model on a new image
predict_segmentation(unet, "test.png")