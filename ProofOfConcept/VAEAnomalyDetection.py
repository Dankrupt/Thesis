import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from VAE_PosDiff import LabeledNoisyMNISTModel, LabeledNoisyMNIST
import os
from tqdm import tqdm
import torch.utils.data as data_utils
from sklearn import cluster, mixture
from itertools import cycle, islice

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()

path = os.path.join(os.getcwd())
model = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-ZERO-PosRotVar-epoch=09.ckpt')
model.freeze()

# Getting the dataset, picking a random entry and adding varying amounts of noise
dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)
indices = torch.arange(0,len(dataset)//2048)
dataset = data_utils.Subset(dataset, indices)

# Define varying levels of noise for each part of the dataset
num_total_images = len(dataset)
noise_levels1 = (np.linspace(0.2, 0.4, num_total_images), 'LOW')
noise_levels2 = (np.linspace(0.4, 0.6, num_total_images), 'MEDIUM')
noise_levels3 = (np.linspace(0.6, 0.8, num_total_images), 'HIGH')

# Shuffle the noise levels to ensure random assignment to images
#np.random.shuffle(noise_levels1)
#np.random.shuffle(noise_levels2)
#np.random.shuffle(noise_levels3)

# Create LabeledNoisyMNIST datasets
noisy_dataset1 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels1)
noisy_dataset2 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels2)
noisy_dataset3 = LabeledNoisyMNIST(dataset, noise_levels=noise_levels3)


loss_function = torch.nn.MSELoss()
loss_normal = []
loss_anomaly1 = []
loss_anomaly2 = []
loss_anomaly3 = []
iter_normal = []
iter_anomaly1 = []
iter_anomaly2 = []
iter_anomaly3 = []

for i, image in tqdm(enumerate(dataset)):
    image = torch.reshape(image[0], (-1, 28*28)).to('cuda')
    reconstructed = model(image)
    loss_normal.append(loss_function(reconstructed[0], image).item())
    iter_normal.append(i)

for i, image in tqdm(enumerate(noisy_dataset1)):
    image = torch.reshape(image[0], (-1, 28*28)).to('cuda')
    reconstructed = model(image)
    loss_anomaly1.append(loss_function(reconstructed[0], image).item())
    iter_anomaly1.append(i)

for i, image in tqdm(enumerate(noisy_dataset2)):
    image = torch.reshape(image[0], (-1, 28*28)).to('cuda')
    reconstructed = model(image)
    loss_anomaly2.append(loss_function(reconstructed[0], image).item())
    iter_anomaly2.append(i)

for i, image in tqdm(enumerate(noisy_dataset3)):
    image = torch.reshape(image[0], (-1, 28*28)).to('cuda')
    reconstructed = model(image)
    loss_anomaly3.append(loss_function(reconstructed[0], image).item())
    iter_anomaly3.append(i)


threshold1 = np.mean(loss_normal) + 1.5*np.std(loss_normal)
threshold2 = np.mean(loss_anomaly1) + 1.5*np.std(loss_anomaly1)
threshold3 = np.mean(loss_anomaly2) + 1.5*np.std(loss_anomaly2)
threshold4 = np.mean(loss_anomaly3) + 1.5*np.std(loss_anomaly3)

spectral = cluster.SpectralClustering(
    n_clusters=4,
    eigen_solver="arpack",
    affinity="nearest_neighbors",
    random_state=42
)

gmm = mixture.GaussianMixture(
    n_components=4,
    covariance_type="full",
    random_state=42,
)

losses = loss_normal + loss_anomaly1 + loss_anomaly2 + loss_anomaly3
iters = iter_normal + iter_anomaly1 + iter_anomaly2 + iter_anomaly3

iters = np.linspace(0, 115, 116)

x = np.random.randint(0, 500, 1000)
y = np.random.randint(0, 500, 1000)
Y = np.array([list(element) for element in list(zip(x, y))])

losses = [loss * 100 for loss in losses]

X = np.array([list(element) for element in list(zip(losses, iters))])

#spectral.fit(X)

y_pred = spectral.fit_predict(X)

colors = np.array(
    list(
        islice(
            cycle(
                [
                    "#377eb8",
                    "#ff7f00",
                    "#4daf4a",
                    "#f781bf",
                    "#a65628",
                    "#984ea3",
                    "#999999",
                    "#e41a1c",
                    "#dede00",
                ]
            ),
            int(max(y_pred) + 1),
        )
    )
)


# Plotting the loss values
with torch.no_grad():
    fig, axs = plt.subplots(1,2)
    #plt.hist(loss_normal, bins=50)
    #plt.hist(loss_anomaly1, bins=50)
    #plt.hist(loss_anomaly2, bins=50)
    #plt.hist(loss_anomaly3, bins=50)
    axs[0].scatter(iters[0: int(len(iters)/4)], loss_normal, s=6)
    axs[0].scatter(iters[int(len(iters)/4): int(len(iters)/2)], loss_anomaly1, s=6)
    axs[0].scatter(iters[int(len(iters)/2): int(3*len(iters)/4)], loss_anomaly2, s=6)
    axs[0].scatter(iters[int(3*len(iters)/4): int(len(iters))], loss_anomaly3, s=6)

    #axs[1].scatter(iters, losses, s=6)

    #plt.axvline(threshold1, color='r', linewidth=2, linestyle='dashed')
    #plt.axvline(threshold2, color='r', linewidth=2, linestyle='dashed')
    #plt.axvline(threshold3, color='r', linewidth=2, linestyle='dashed')
    #plt.axvline(threshold4, color='r', linewidth=2, linestyle='dashed')
    #plt.title('1 Model')

    axs[1].scatter(X[:, 1], X[:, 0], s=6, color=colors[y_pred])
    plt.show()