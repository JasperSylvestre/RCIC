"""
model_prediction.py

Originally built in Google colab.
"""

# !mkdir ../data/test/test_images
# !mv ../*jpg ../data/test/test_images

# !pip install dill

import torch
import dill
import os
from PIL import Image
from torchvision import transforms
import time

# Create necessary directories and enter scripts folder
if os.path.basename(os.getcwd()) != 'scripts':
  os.makedirs('scripts', exist_ok=True)
  os.chdir('scripts')
os.makedirs('../models', exist_ok=True)
os.makedirs('../data/test/test_images', exist_ok=True)

# Read in final models
model_dict = {}
models = ['mobilenetv2', 'resnet50', 'vgg16']

for model_name in models:
  with open(f'../models/final-{model_name}.dill', 'rb') as f:
    model_dict[model_name] = dill.load(f)

# Define data transformer
transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),      #  gray-scale -> RGB
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   #  Imagenet means/standard deviation
])

# Record the start time for pre-processing
start_preprocessing_time = time.time()

# Load the images from the test images folder and apply transformations
test_images = []
for filename in os.listdir('../data/test/test_images'):
  if filename.endswith('.jpg'):
    img_path = os.path.join('../data/test/test_images', filename)
    img = Image.open(img_path)
    img = transform(img)
    test_images.append(img)

# Convert the list of images to a tensor
test_images_tensor = torch.stack(test_images)

# Record the end time for pre-processing
end_preprocessing_time = time.time()

preprocessing_time = end_preprocessing_time - start_preprocessing_time
print(f"Pre-processing time: {preprocessing_time:.2f} seconds")

# Define the batch size
batch_size = 32

# Dictionary to store predictions and prediction times for each model
predictions_dict = {}

# Loop through each model
for model_name, model in model_dict.items():
  # Set the model to evaluation mode
  model.eval()

  # Initialize predictions list
  predictions = []

  # Record the start time for making predictions
  start_prediction_time = time.time()

  # Perform prediction with batch size
  with torch.no_grad():
    for i in range(0, len(test_images_tensor), batch_size):
      batch = test_images_tensor[i:i+batch_size]
      batch_outputs = model(batch)
      batch_predictions = torch.argmax(batch_outputs, dim=1)
      predictions.extend(batch_predictions.tolist())

  # Calculate prediction time
  prediction_time = time.time() - start_prediction_time

  # Store predictions and prediction time in dictionary
  predictions_dict[model_name] = {
    'predictions': predictions,
    'prediction_time': prediction_time
  }

# Print prediction time for each model
for model_name, results in predictions_dict.items():
  print(f"Prediction time for {model_name}: {results['prediction_time']:.2f} seconds")
