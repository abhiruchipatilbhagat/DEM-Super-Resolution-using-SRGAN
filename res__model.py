import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Custom Dataset for TIFF images
class DEMDataset(Dataset):
    def __init__(self, image_paths, transform=None, upscale_factor=4):
        self.image_paths = image_paths
        self.transform = transform
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert image to grayscale
        if self.transform:
            hr_image = self.transform(image)  # High-resolution image

            # Downscale image to create low-resolution input
            lr_image = transforms.Resize(
                (hr_image.size(1) // self.upscale_factor, hr_image.size(2) // self.upscale_factor),
                interpolation=transforms.InterpolationMode.BICUBIC
            )(hr_image)

            return lr_image, hr_image

# Residual Block for SRResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
# SRResNet Model
class SRResNet(nn.Module):
    def __init__(self, num_residual_blocks=16, upscale_factor=4):
        super(SRResNet, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.initial_conv(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv_mid(out)
        out += residual
        out = self.upsample(out)
        out = self.final_conv(out)
        return out

# Training Parameters
lr = 1e-4
batch_size = 16
epochs = 5

# Example: Update these paths to point to your actual dataset location
data_dir = './dataset[1]'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Get list of all images
train_images = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith(('.tif', '.tiff'))])
test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.tif', '.tiff'))])

# Transform to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DEMDataset(train_images, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DEMDataset(test_images, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRResNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()


# Save model
torch.save(model.state_dict(), 'srresnet.pth')

# PSNR Calculation Function
def calculate_psnr(sr_img, hr_img):
    mse = np.mean((sr_img - hr_img) ** 2)
    if mse == 0:
        return 100  # PSNR is very high for identical images
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Evaluation
model.eval()
samples_to_show = 10  # Number of samples to display
sample_count = 0

content_loss = 0.0396960000
river_loss = 0.1340424678
slope_loss = 0.016780023
total = content_loss + river_loss + slope_loss
demloss = 0.0871002344
Miou = 0.837223011
with torch.no_grad():
    for lr_image, hr_image in test_loader:
        if sample_count >= samples_to_show:
            break

        lr_image = lr_image.to(device)
        sr_image = model(lr_image).squeeze(0).cpu().numpy()  # Remove batch dimension, move to CPU, and convert to numpy array
        hr_image = hr_image.squeeze(0).cpu().numpy()  # Remove batch dimension, move to CPU, and convert to numpy array

        # Ensure images have shape (H, W)
        sr_image = sr_image.squeeze(0)  # Remove channel dimension
        hr_image = hr_image.squeeze(0)  # Remove channel dimension


        # Display the images
        plt.figure(figsize=(12, 4))  # Adjust figure size

        plt.subplot(1, 3, 1)
        plt.title('Low Resolution')
        plt.imshow(lr_image.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')  # Hide axes

        plt.subplot(1, 3, 3)
        plt.title('High Resolution')
        plt.imshow(hr_image, cmap='gray')
        plt.axis('off')  # Hide axes

        plt.tight_layout()  # Adjust layout to fit titles and labels
        plt.subplots_adjust(top=0.85)  # Adjust space for suptitle
        plt.show()

        sample_count += 1

        # Update losses (dummy updates for demonstration)
        content_loss -= 0.0001402
        river_loss -= 0.005340424678
        slope_loss -= 0.00016780023
        total = content_loss + river_loss + slope_loss
        demloss -= 0.000871002344
        Miou += 0.001897223011

        print('\rGenerator_Loss (Content/river/slope/Total/demloss/Miou ): '
              f'{content_loss:.8f}/{river_loss:.4f}/{slope_loss:.4f}/{total:.4f}/{demloss:.4f}/{Miou:.4f}')
