import argparse
import os
import sys

sys.path.append("..")
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage import io
import random
import cv2
import time
######################################################
from DeviceSetting import device
from tfasr_model import Generator as srnet
from FeatureLoss import river_conterion, unet
from DEM_features import Slope_net

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the high resolution image size')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataroot', type=str,
                    default='dataset',
                    help='path to dataset')
parser.add_argument('--netweight', type=str,
                    default='',
                    help="path to generator weights (to continue training)")
out = "tfasr_" + str(1)
parser.add_argument('--river_weight', type=float, default=1, help='')  # α
parser.add_argument('--slope_weight', type=float, default=0, help='')  # β
parser.add_argument('--out', type=str, default='./checkpoints/' + out,
                    help='folder to output model checkpoints')
parser.add_argument('--logfile', default='./errlog-' + out + '.txt',
                    help="pre-training epoch times")
parser.add_argument('--ave_logfile', default='./ave-' + out + '.txt',
                    help="pre-training epoch times")

opt = parser.parse_args()

print(opt)

# output directory creation
try:
    os.makedirs(opt.out)
except OSError:
    pass


def write_file(filepath, target_tensor):
    with open(filepath, 'a') as af:
        num_rows, num_cols = target_tensor.shape
        for i in range(num_rows):
            for j in range(num_cols):
                af.write(str(target_tensor[i][j].item()) + ',')
            af.write('\n')


dataset = datasets.ImageFolder(root=opt.dataroot)
assert dataset

def check_dataset(dataroot):
    train_folder = os.path.join(dataroot, 'train')
    test_folder = os.path.join(dataroot, 'test')
    
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f'Train folder not found at {train_folder}')
    
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f'Test folder not found at {test_folder}')
    
    print("Train and test folders found.")

    train_dataset = datasets.ImageFolder(root=train_folder)
    test_dataset = datasets.ImageFolder(root=test_folder)

    train_classes = train_dataset.classes
    test_classes = test_dataset.classes

    print(f"Train classes: {train_classes}")
    print(f"Test classes: {test_classes}")

    train_class_counts = {cls: 0 for cls in train_classes}
    test_class_counts = {cls: 0 for cls in test_classes}

    for _, label in train_dataset.samples:
        train_class_counts[train_dataset.classes[label]] += 1

    for _, label in test_dataset.samples:
        test_class_counts[test_dataset.classes[label]] += 1

    print(f"Train class counts: {train_class_counts}")
    print(f"Test class counts: {test_class_counts}")

    return train_dataset, test_dataset

train_dataset, test_dataset = check_dataset(opt.dataroot)

for idx in range(5):
    img_path, _ = dataset.imgs[idx]
    print(f"Image path: {img_path}")
    img_temp = io.imread(img_path)
    print(f"Image shape: {img_temp.shape}")

model = srnet(16, opt.upSampling).to(device)

if opt.netweight != '':
    model.load_state_dict(torch.load(opt.netweight))
#print(model)

optimiser = optim.Adam(model.parameters(), lr=opt.lr)
content_criterion = nn.MSELoss().to(device)

high_res_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # high resolution dem
low_res = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # low resolution dem
base_min_list = torch.FloatTensor(opt.batchSize, 1).to(device)
base_max_list = torch.FloatTensor(opt.batchSize, 1).to(device)
high_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # river
fake_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
original_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # dem
fake_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
high_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
fake_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)

imgs_count = len(dataset.imgs)  # training dems count
print(imgs_count)
batchSize_count = imgs_count // opt.batchSize  # number of batchSize in each epoch

model.train()
random_ids = list(range(0, imgs_count))
random.shuffle(random_ids)  # shuffle these dems
print('river training')
for epoch in range(0, opt.nEpochs):
    if epoch >= 80:
        opt.river_weight = 1e-3
    mean_generator_content_loss = 0.0
    mean_generator_river_loss = 0.0
    mean_generator_slope_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_dem_loss = 0.0
    mean_miou_0 = 0.0
    mean_miou_1 = 0.0
    mean_Miou = 0.0
    for i in range(batchSize_count):
        # get a batchsize of rivers and dems
        for j in range(opt.batchSize):
            img_temp, _ = dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution image
            img_temp = io.imread(img_temp)
            img_temp = cv2.resize(img_temp, (opt.imageSize, opt.imageSize))
            H, W = img_temp.shape
            # low-resolution image
            low_img_temp = cv2.resize(img_temp, (H // opt.upSampling, W // opt.upSampling),
                                      interpolation=cv2.INTER_NEAREST)
            base_min = np.min(low_img_temp)
            base_max = np.max(low_img_temp)
            base_min_list[j] = torch.tensor(base_min)
            base_max_list[j] = torch.tensor(base_max)
            bicubic_high_img_temp = cv2.resize(low_img_temp, (H, W), interpolation=cv2.INTER_NEAREST)
            img_temp = torch.tensor(img_temp).unsqueeze(0)  # 1*imagesize*imagesize
            #print(img_temp.shape)
            original_dem[j] = img_temp
            # 10 is a default value to keep safe
            img_temp = 2 * (img_temp - base_min) / (base_max - base_min + 10) - 1
            bicubic_high_img_temp = 2 * (bicubic_high_img_temp - base_min) / (base_max - base_min + 10) - 1
            high_res_real[j] = img_temp  # -1~1
            low_res[j] = torch.tensor(bicubic_high_img_temp)

        high_river = 1.0 * (F.sigmoid(unet(high_res_real.to(device))).detach().cpu() > 0.5).numpy().astype(np.float32)
        high_river_heatmap = torch.tensor(high_river).to(device)

        # Generate real and fake inputs
        high_res_real = Variable(high_res_real.to(device))
        high_res_fake = model(low_res.to(device)).to(device)

        ######### Train generator #########
        optimiser.zero_grad()

        high_slope = Slope_net(high_res_real)
        fake_slope = Slope_net(high_res_fake)
        generator_slope_loss = content_criterion(high_slope, fake_slope)

        generator_content_loss = content_criterion(high_res_fake.to(device), high_res_real.to(device))

        generator_river_loss, miou_0, miou_1, Miou = river_conterion(high_res_fake.to(device),
                                                                     high_river_heatmap.to(device),
                                                                     base_max_list.to(device))

        generator_total_loss = generator_content_loss + opt.slope_weight * generator_slope_loss + opt.river_weight * generator_river_loss

        generator_total_loss.backward()
        optimiser.step()

        dem_loss = 0
        for j in range(opt.batchSize):
            fake_dem[j] = (0.5 * (high_res_fake[j] + 1) * (base_max_list[j] - base_min_list[j] + 10) +
                           base_min_list[j]).to(device)
            dem_loss += math.sqrt(content_criterion(original_dem[j], fake_dem[j]))
        dem_loss = dem_loss / opt.batchSize
        ######### Status and display #########
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, batchSize_count,
                generator_content_loss.data,
                generator_river_loss.data,
                generator_slope_loss.data,
                generator_total_loss.data,
                dem_loss, miou_0, miou_1, Miou))
        errlog = open(opt.logfile, 'a')
        errlog.write(
            '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, batchSize_count,
                generator_content_loss.data,
                generator_river_loss.data,
                generator_slope_loss.data,
                generator_total_loss.data,
                dem_loss, miou_0, miou_1, Miou))
        errlog.close()
        ######## mean value of each epoch ##############
        mean_generator_content_loss += generator_content_loss.data
        mean_generator_river_loss += generator_river_loss.data
        mean_generator_slope_loss += generator_slope_loss.data
        mean_generator_total_loss += generator_total_loss.data
        mean_dem_loss += dem_loss
        mean_miou_0 += miou_0
        mean_miou_1 += miou_1
        mean_Miou += Miou

    mean_generator_content_loss = mean_generator_content_loss / batchSize_count
    mean_generator_river_loss = mean_generator_river_loss / batchSize_count
    mean_generator_slope_loss = mean_generator_slope_loss / batchSize_count
    mean_generator_total_loss = mean_generator_total_loss / batchSize_count
    mean_dem_loss = mean_dem_loss / batchSize_count
    mean_miou_0 = mean_miou_0 / batchSize_count
    mean_miou_1 = mean_miou_1 / batchSize_count
    mean_Miou = mean_Miou / batchSize_count

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, batchSize_count,
            mean_generator_content_loss, mean_generator_river_loss, generator_slope_loss,
            mean_generator_total_loss,
            mean_dem_loss, mean_miou_0, mean_miou_1, mean_Miou))
    ave_errlog = open(opt.ave_logfile, 'a')
    ave_errlog.write(
        '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, batchSize_count,
            mean_generator_content_loss, mean_generator_river_loss, generator_slope_loss,
            mean_generator_total_loss,
            mean_dem_loss, mean_miou_0, mean_miou_1, mean_Miou))
    ave_errlog.close()

    # Do checkpointing
    torch.save(model.state_dict(), '%s/generator_final_%03d.pth' % (opt.out, epoch))
