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
model1 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-ZERO-epoch=19.ckpt')
model2 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-LOW-epoch=19.ckpt')
model3 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-MEDIUM-epoch=19.ckpt')
model4 = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-HIGH-epoch=19.ckpt')
model1.freeze()
model2.freeze()
model3.freeze()
model4.freeze()

# Getting the dataset, picking a random entry and adding varying amounts of noise
dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)
indices = torch.arange(0,len(dataset)//128)
dataset = data_utils.Subset(dataset, indices)

# Define varying levels of noise for each part of the dataset
num_total_images = len(dataset)
num_images_per_part = num_total_images // 4
noise_levels_part1 = np.linspace(0.0, 0.2, num_images_per_part)
noise_levels_part2 = np.linspace(0.2, 0.4, num_images_per_part)
noise_levels_part3 = np.linspace(0.4, 0.6, num_images_per_part)
noise_levels_part4 = np.linspace(0.6, 0.8, num_images_per_part)

# Combine noise levels for the entire dataset
all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

# Shuffle the noise levels to ensure random assignment to images
#np.random.shuffle(all_noise_levels)

# Create LabeledNoisyMNIST datasets
noisy_dataset = LabeledNoisyMNIST(dataset, noise_levels=all_noise_levels)

#loss_function = LabeledNoisyMNISTModel.loss_function()
loss_function = torch.nn.MSELoss()
error1, error2, error3, error4 = ([] for i in range(4))
for image in tqdm(noisy_dataset):
    image = torch.reshape(image[0], (-1, 28*28))

    reconstructed1 = model1(image)
    error1.append(loss_function(reconstructed1[0], image))

    reconstructed2 = model2(image)
    error2.append(loss_function(reconstructed2[0], image))

    reconstructed3 = model3(image)
    error3.append(loss_function(reconstructed3[0], image))

    reconstructed4 = model4(image)
    error4.append(loss_function(reconstructed4[0], image))

window = 20
average_error1, average_error2, average_error3, average_error4 = ([] for i in range(4))
for i in tqdm(range(len(dataset) - window + 1)):
    average_error1.append(np.mean(error1[i:i+window]))
    average_error2.append(np.mean(error2[i:i+window]))
    average_error3.append(np.mean(error3[i:i+window]))
    average_error4.append(np.mean(error4[i:i+window]))

for ind in tqdm(range(window - 1)):
    average_error1.insert(0, np.nan)
    average_error2.insert(0, np.nan)
    average_error3.insert(0, np.nan)
    average_error4.insert(0, np.nan)

#for ind in tqdm(range(window - 1)):
#    error1.insert(0, np.nan)
#    error2.insert(0, np.nan)
#    error3.insert(0, np.nan)
#    error4.insert(0, np.nan)

# Plotting the loss values
with torch.no_grad():
    plt.figure()
    plt.plot(average_error1, label='ZERO')
    plt.plot(average_error2, label='LOW')
    plt.plot(average_error3, label='MEDIUM')
    plt.plot(average_error4, label='HIGH')
    plt.plot(error1, '--', label='ZERO', linewidth=0.1)
    plt.plot(error2, '--', label='LOW', linewidth=0.1)
    plt.plot(error3, '--', label='MEDIUM', linewidth=0.1)
    plt.plot(error4, '--', label='HIGH', linewidth=0.1)
    plt.legend()
    plt.show()