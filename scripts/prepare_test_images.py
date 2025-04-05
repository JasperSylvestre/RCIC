"""
prepare_test_images.ipynb

Originally built in Google colab.
"""

# !pip install dill

import dill
import numpy as np
import os
from PIL import Image

# Create necessary directories and enter scripts folder
if os.path.basename(os.getcwd()) != 'scripts':
  os.makedirs('scripts', exist_ok=True)
  os.chdir('scripts')
os.makedirs('../models', exist_ok=True)
os.makedirs('../data/test/test_images', exist_ok=True)

# Load the saved validation indices and the original dataset
with open('../data/test/test_ds.pkl', 'rb') as f:
  test_ds = dill.load(f)

# Unnormalize from Imagenet mean/standard deviation
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

original_ds = (test_ds * imagenet_std[:, None, None]) + imagenet_mean[:, None, None]

# Loop through the array and save each image as a JPG
for i in range(original_ds.shape[0]):
  img = original_ds[i].transpose(1, 2, 0)
  img = Image.fromarray((img * 255).astype(np.uint8))  # Normalize to [0, 255]
  img.save(os.path.join(output_dir, f'{i:03d}.jpg'))
