import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import cv2

# ML LIBRARIES
from keras.utils import plot_model
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import legacy

# AUGMENTING THE DATASET
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img


root_path = r"D:\Whatever in stock\test\BRAIN SCAN IMAGES\TARP\datasets"

no_path , yes_path = os.path.join(root_path , 'NO') , os.path.join(root_path , 'YES')

def convert_to_grayscale(folder_path):
    # Iterate through all files in the folder
    cnt = 1
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(file_path)
            
            # Convert to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            file_new_path = os.path.join(folder_path, "test_no" + str(cnt) + ".jpg")
            
            # Write the grayscale image back to the same path
            cv2.imwrite(file_new_path, grayscale_image)
            os.remove(file_path)
            print(f"Converted {file_new_path} to grayscale.")
        else:
            print(f"Skipping non-image file: {file_new_path}")
        
        cnt += 1
            
convert_to_grayscale(r"D:\Whatever in stock\test\BRAIN SCAN IMAGES\TARP\datasets\test\no")