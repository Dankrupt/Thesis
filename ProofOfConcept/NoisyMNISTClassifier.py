import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from NoisyMNIST import LabeledNoisyMNISTModel
import os


# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()

path = os.path.join(os.getcwd())
model = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-epoch=09-train_loss=0.05.ckpt')
model.freeze()

# Getting the dataset, picking a random entry and adding varying amounts of noise
dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)
image = torch.reshape(random.choice(dataset)[0], (-1, 28*28))
noise_levels = [(0, 'zero'), (0.4, 'low'), (0.8, 'high'), (0.6, 'medium')]
images_noise = [torch.clamp(image + level[0] * torch.rand_like(image), 0, 1) for level in noise_levels]

# Output of Autoencoder
reconstructed = [model(image) for image in images_noise]

# Calculating the loss function
loss_function = torch.nn.MSELoss()
losses = [loss_function(reconstructed, images_noise[i]) for i, reconstructed in enumerate(reconstructed)]

# Plotting the loss values
with torch.no_grad():
    plt.figure()
    plt.scatter([level[1] for level in noise_levels], losses)
    plt.show()