import subprocess
import sys

import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np

# Import your model and other required modules
from model import Generator as srnet
from DeviceSetting import device

# Define argument parser for input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--imagePath', type=str, help='path to the input image')
parser.add_argument('--netweight', type=str, help="path to the generator weights")
parser.add_argument('--imageSize', type=int, default=64, help='the high resolution image size')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
opt = parser.parse_args()

# Load the model
model = srnet(16, opt.upSampling).to(device)
#model.load_state_dict(torch.load(opt.netweight, map_location=torch.device('cpu')))
#model.eval()

# Function to process and display images
def display_images(original_img, low_res_img, high_res_img):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')

    axes[1].imshow(low_res_img, cmap='gray')
    axes[1].set_title('Low Resolution')
    axes[1].axis('off')

    axes[2].imshow(high_res_img, cmap='gray')
    axes[2].set_title('Super Resolved')
    axes[2].axis('off')

    plt.show()

def display__images(script_name):
    try:
        # Run the specified script
        result = subprocess.run([sys.executable, script_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"\n{result.stdout.decode('utf-8')}")
        print(f"Script completed with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")

def read():# Read and preprocess the image
    original_img = io.imread(opt.imagePath).astype(np.float32)
    H, W = original_img.shape
    low_res_img = cv2.resize(original_img, (H // opt.upSampling, W // opt.upSampling), interpolation=cv2.INTER_NEAREST)
    low_res_img_bicubic = cv2.resize(low_res_img, (H, W), interpolation=cv2.INTER_CUBIC)

# Normalize the images to -1 to 1 range
    base_min = np.min(low_res_img)
    base_max = np.max(low_res_img)
    original_img_norm = 2 * (original_img - base_min) / (base_max - base_min + 10) - 1
    low_res_img_bicubic_norm = 2 * (low_res_img_bicubic - base_min) / (base_max - base_min + 10) - 1

    # Convert to torch tensors
    original_img_tensor = torch.tensor(original_img_norm).unsqueeze(0).unsqueeze(0).to(device)
    low_res_img_tensor = torch.tensor(low_res_img_bicubic_norm).unsqueeze(0).unsqueeze(0).to(device)

def display__images(script_name):
    try:
        # Run the specified script
        result = subprocess.run([sys.executable, script_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"\n{result.stdout.decode('utf-8')}")
        print(f"Script completed with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")

# Generate high resolution image using the model


if __name__ == "__main__":
    # Update this with the path to your res__model.py script
    script_path = "res__model.py"
    display__images(script_path)
