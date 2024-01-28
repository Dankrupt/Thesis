import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from VAE import LabeledNoisyMNISTModel, LabeledNoisyMNIST
import os
from tqdm import tqdm
import torch.utils.data as data_utils

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()

path = os.path.join(os.getcwd())
model = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-ZERO-epoch=19.ckpt')
model2 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-LOW-epoch=19.ckpt')
model3 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-MEDIUM-epoch=19.ckpt')
model4 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-HIGH-epoch=19.ckpt')
model.freeze()
model2.freeze()
model3.freeze()
model4.freeze()

# Getting the dataset, picking a random entry and adding varying amounts of noise
dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)
indices = torch.arange(0,len(dataset)//32)
dataset = data_utils.Subset(dataset, indices)

# Define varying levels of noise for each part of the dataset
num_total_images = len(dataset)
noise_levels1 = np.linspace(0.2, 0.4, num_total_images)
noise_levels2 = np.linspace(0.4, 0.6, num_total_images)
noise_levels3 = np.linspace(0.6, 0.8, num_total_images)

# Shuffle the noise levels to ensure random assignment to images
np.random.shuffle(noise_levels1)
np.random.shuffle(noise_levels2)
np.random.shuffle(noise_levels3)

# Create LabeledNoisyMNIST datasets
noisy_dataset1 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels1)
noisy_dataset2 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels2)
noisy_dataset3 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels3)

loss_function = torch.nn.MSELoss()
loss_normal = []
loss_anomaly1 = []
loss_anomaly2 = []
loss_anomaly3 = []

for image in tqdm(dataset):
    image = torch.reshape(image[0], (-1, 28*28))

    reconstructed = model(image)
    loss_normal.append(loss_function(reconstructed[0], image))

for image in tqdm(noisy_dataset1):
    image = torch.reshape(image[0], (-1, 28*28))

    reconstructed = model2(image)
    loss_anomaly1.append(loss_function(reconstructed[0], image))

for image in tqdm(noisy_dataset2):
    image = torch.reshape(image[0], (-1, 28*28))

    reconstructed = model3(image)
    loss_anomaly2.append(loss_function(reconstructed[0], image))

for image in tqdm(noisy_dataset3):
    image = torch.reshape(image[0], (-1, 28*28))

    reconstructed = model4(image)
    loss_anomaly3.append(loss_function(reconstructed[0], image))


threshold1 = np.mean(loss_normal) + 1.5*np.std(loss_normal)
threshold2 = np.mean(loss_anomaly1) + 1.5*np.std(loss_anomaly1)
threshold3 = np.mean(loss_anomaly2) + 1.5*np.std(loss_anomaly2)
threshold4 = np.mean(loss_anomaly3) + 1.5*np.std(loss_anomaly3)

# Plotting the loss values
with torch.no_grad():
    plt.figure()
    plt.hist(loss_normal, bins=50)
    plt.hist(loss_anomaly1, bins=50)
    plt.hist(loss_anomaly2, bins=50)
    plt.hist(loss_anomaly3, bins=50)
    plt.axvline(threshold1, color='r', linewidth=2, linestyle='dashed')
    plt.axvline(threshold2, color='r', linewidth=2, linestyle='dashed')
    plt.axvline(threshold3, color='r', linewidth=2, linestyle='dashed')
    plt.axvline(threshold4, color='r', linewidth=2, linestyle='dashed')
    plt.title('4 Models')
    plt.show()