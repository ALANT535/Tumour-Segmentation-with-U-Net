from ultralytics import SAM
import cv2
import matplotlib.pyplot as plt

# Load the model
model = SAM('mobile_sam.pt')

image = cv2.imread(r'D:\Whatever in stock\test\BRAIN SCAN IMAGES\TARP\datasets\test\yes\test_yes1.jpg')

# Run automatic segmentation
results = model.predict(image, mode='automatic')  # `top_n` limits to top 5 objects

# print(results)

for i, mask in enumerate(results):  # Assume `masks` is a list of binary masks
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask {i+1}')
    plt.axis('off')
    plt.show()