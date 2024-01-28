import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torchvision.transforms import v2

#tensor_transform = transforms.ToTensor()
tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=180, translate=(0.25,0.25)),
        transforms.ConvertImageDtype(float)
    ]
)

tensor_transform = v2.RandomAffine(degrees=180,translate=(0.4, 0.4),scale=(0.5,0.5))

tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        v2.RandomAffine(degrees=180,translate=(0.4, 0.4),scale=(0.5,0.5)),
    ]
)

dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)

#print(dataset[0][0][0].shape)
#print(dataset[1][0])

#plt.imshow(np.asarray(dataset[1][0]))
#plt.show()

fig, axs = plt.subplots(2,3)
axs[0,0].imshow(np.asarray(dataset[0][0][0]))
axs[0,1].imshow(np.asarray(dataset[1][0][0]))
axs[0,2].imshow(np.asarray(dataset[2][0][0]))
axs[1,0].imshow(np.asarray(dataset[3][0][0]))
axs[1,1].imshow(np.asarray(dataset[4][0][0]))
axs[1,2].imshow(np.asarray(dataset[5][0][0]))
plt.show()

"""
background = np.zeros((56,56))
topLeft = (random.randint(0,27), random.randint(0,27))
topLeft = (0,0)
i = 0
for image in dataset:
    #print(image[0][0])
    image = image[0][0].numpy()
    background[topLeft[0]:topLeft[0]+28, topLeft[1]:topLeft[1]+28] = image[0][0]
    #background = torch.from_numpy(background)

    dataset[i][0][0] = background
    i += 1

    plt.imshow(background)

dataset = background
plt.imshow(dataset[0][0])
plt.show()
"""