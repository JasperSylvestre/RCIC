"""
train_model.ipynb

Originally built in Google colab.
"""

# !pip install dill

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import os
import dill
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

# Create and enter scripts directory if not present
if os.path.basename(os.getcwd()) != 'scripts':
  if not os.path.exists('scripts'):
    os.makedirs('scripts')
  os.chdir('scripts')

# Create models directory
if not os.path.exists('../models'):
  os.makedirs('../models')

# Create data/test directory
if not os.path.exists('../data/test'):
  os.makedirs('../data/test')

# Define data transformer
transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),      #  gray-scale -> RGB
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   #  Imagenet means/standard deviation
])

# Download and load the Caltech-101 dataset
ds = torchvision.datasets.Caltech101(root='../data', download=True, transform=transform)

# Stratified splitting into training and testing data sets
RANDOM_SEED = 123  # Set seed for reproducability

train_indices, val_indeces = train_test_split(range(len(ds)), test_size=0.2, stratify=[ds[i][1] for i in range(len(ds))], random_state=RANDOM_SEED)

train_ds = torch.utils.data.Subset(ds, train_indices)
val_ds = torch.utils.data.Subset(ds, val_indeces)

# Save a subset of the validation dataset's image arrays to prepare in prepare_test_images.py
val_labels = [label for _, label in val_ds]
_, test_indices = train_test_split(range(len(val_ds)), test_size=101, stratify=val_labels, random_state=RANDOM_SEED)

test_ds_array = np.array([image.numpy() for image, label in torch.utils.data.Subset(val_ds, test_indices)])

with open('../data/test/test_ds.pkl', 'wb') as f:
  dill.dump(test_ds_array, f)

# Define the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
NUM_CLASSES = 101

# Create a dictionary of models
model_dict = {
  'mobilenetv2': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
  'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
  'vgg16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
}

# Modify each model in the dictionary
for name, model in model_dict.items():
  # Freeze the model's pre-trained weights
  for param in model.parameters():
    param.requires_grad = False

    # Replace the last layer for prediction on Caltech-101
    if name == 'mobilenetv2':
      model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    elif name == 'resnet50':
      model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == 'vgg16':
      model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, NUM_CLASSES)

    # Move the model to the device
    model_dict[name] = model.to(device)

# Create data loaders
BATCH_SIZE = 32

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Define the loss function and optimizer
LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss()

def train_model(model_name, model, train_loader, val_loader, optimizer, criterion, device, epochs, patience):
  """
  Train a PyTorch model on a set of dataloaders.

  Args:
    model_name (str): The name of model for the save's filepath.
    model (nn.Module): The model to train.
    train_loader (DataLoader): The training data loader.
    val_loader (DataLoader): The validation data loader.
    optimizer (Optimizer): The optimizer to use.
    criterion (nn.Module): The loss function to use.
    device (torch.device): The device to train on.
    epochs (int): The number of epochs to train for.
    patience (int): The number of epochs to wait for improvement before stopping.
  """
  model.to(device)

  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    # Iterate over training batches
    for i, (inputs, labels) in enumerate(train_loader, 0):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    # Evaluate model on validation data
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss

    print(f'Epoch {epoch + 1} | Training loss: {train_loss:.2f} | Validation loss: {val_loss:.2f}')

    # Check for early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save(model.state_dict(), f'../models/{model_name}.pth')  # Save best model weights
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

def evaluate_model(model, val_ds, device, batch_size):
  """
  Evaluate a PyTorch model on a validation dataset.

  Args:
    model (nn.Module): The model to evaluate.
    val_ds (Dataset): The validation dataset.
    device (torch.device): The device to use.
    batch_size (int): The batch size.

  Returns:
    float: The accuracy of the model on the validation dataset.
  """
  # Initialize accuracy metrics
  total_correct = 0
  total_samples = 0

  # Set the model to evaluation mode
  model.eval()

  # Disable gradient computation
  with torch.no_grad():
    # Create a data loader for the validation dataset
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Make predictions on the validation dataset
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)

      # Update accuracy metrics
      total_correct += (predicted == labels).sum().item()
      total_samples += labels.shape[0]

  # Calculate accuracy
  accuracy = total_correct / total_samples

  return accuracy

EPOCHS = 30  #  30
PATIENCE_VALUE = 5

# Train models
for name, model in model_dict.items():
  print(f'{name.title()} model training:')

  train_model(
    model_name=name.upper(),
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
    criterion=criterion,
    device=device,
    epochs=EPOCHS,
    patience=PATIENCE_VALUE
  )

  print()

# Create a dictionary of the final models
model_dict_final = {
  'mobilenetv2': torchvision.models.mobilenet_v2(weights=None),
  'resnet50': torchvision.models.resnet50(weights=None),
  'vgg16': torchvision.models.vgg16(weights=None),
}

# Load models
for name in model_dict_final.keys():
  model = model_dict_final[name]

  if name == 'mobilenetv2':
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
  elif name == 'resnet50':
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
  elif name == 'vgg16':
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, NUM_CLASSES)

  model.load_state_dict(torch.load(f'../models/{name.upper()}.pth'))
  model_dict[name] = model.to(device)

# Evaluate final models
for name, model in model_dict_final.items():
  accuracy = evaluate_model(
    model=model,
    val_ds=val_ds,
    device=device,
    batch_size=BATCH_SIZE
  )

  print(f'{name.title()} Accuracy: {100 * accuracy:.2f}%')

# Save final models
for name, model in model_dict_final.items():
  with open(f'../models/final-{name}.dill', 'wb') as f:
    dill.dump(model.to('cpu'), f)