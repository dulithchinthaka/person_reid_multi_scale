import torch
import torchvision.models as models
import os

# Step 1: Load the pretrained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Step 2: Define the directory where you want to save the model
save_directory = '/media/dulith/bc5d61db-b54c-434d-b2d4-6a104514a052/dulith/Data/MSc/PANet/person_reid_multi_scale/output'  # Replace with your desired directory path

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Step 3: Define the file path
model_path = os.path.join(save_directory, 'resnet50_pretrained.pth')

# Step 4: Save the model's state dictionary
torch.save(resnet50.state_dict(), model_path)

print(f"Model saved to {model_path}")
