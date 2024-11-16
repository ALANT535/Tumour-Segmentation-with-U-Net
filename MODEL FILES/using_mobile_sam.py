from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import torch
import matplotlib.pyplot as plt
import sys

# Load the model
model_type = "vit_t"
sam_checkpoint = "mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# Load your MRI scan image
image_path = "path_to_your_mri_scan.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Use automatic segmentation
mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(image)

# Visualize masks
for mask in masks:
    plt.imshow(mask['segmentation'], cmap='gray')
    plt.axis('off')
    plt.show()
