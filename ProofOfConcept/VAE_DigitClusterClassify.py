import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from VAE import LabeledNoisyMNISTDataModule
from VAE import LabeledNoisyMNISTModel, LabeledNoisyMNIST
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np


# Classification
class FullyConnectedClassifier(nn.Module):
    def __init__(self):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # Flatten 28x28 images to 784-dimensional vectors
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # Output layer: 10 units for 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation; this will be handled by nn.CrossEntropyLoss
        return x


# Initialize the model and data
tensor_transform = transforms.ToTensor()
path = os.path.join(os.getcwd())
model = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-ZERO-Clean-epoch=09.ckpt')
model.eval()  # Set the model to evaluation mode
dataset = datasets.MNIST(root='./data', train=True, transform=tensor_transform)
#indices = torch.arange(0,11000)
#dataset = data_utils.Subset(dataset, indices)
#mnist_train, mnist_val = random_split(dataset, [10000, 1000])

# Take a subset of the dataset for quicker processing
subset_size = 10000
subset_indices = torch.randperm(len(dataset))[:subset_size]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocess and encode images
encoded_images_subset = []
true_labels = []
for idx in subset_indices:
    img, label = dataset[idx]
    img = img.view(-1, 28*28).to(device)  # Flatten image and send to device
    mu, log_var = model.encode(img.unsqueeze(0))  # Add batch dimension
    
    # Ensure mu is squeezed correctly, removing unnecessary dimensions
    mu = mu.squeeze()  # This should adjust mu to have shape [100] or similar
    if mu.dim() == 1:  # If mu is correctly shaped as a 1D tensor
        encoded_images_subset.append((mu, log_var))
        true_labels.append(label)
    else:
        print("Unexpected shape encountered for mu:", mu.shape)

# Concatenate mu values and remove extra dimensions
mu_values = torch.stack([mu for mu, _ in encoded_images_subset])

# Ensure mu_values is a 2D array before clustering
mu_values_2d = mu_values.cpu().detach().numpy()  # Convert to NumPy array for KMeans

# Use KMeans to cluster the encoded images
kmeans = KMeans(n_clusters=10, random_state=0)
spectral = SpectralClustering(n_clusters=10, random_state=0)
gmm = GaussianMixture(n_components=10, random_state=10)
miniBatchKMeans = MiniBatchKMeans(n_clusters=10, random_state=10)
algorithm = kmeans
algorithm.fit(mu_values_2d)

# Reduce dimensions of mu_values to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=0)
#mu_values_2d_tsne = tsne.fit_transform(mu_values_2d.reshape(-1, 1))
mu_values_2d_tsne = tsne.fit_transform(mu_values_2d)

# Assign cluster labels for classification
label_mapping = {}
for cluster in set(algorithm.labels_):
    # Find all true labels for data points in this cluster
    labels_in_cluster = [true_labels[i] for i in range(len(true_labels)) if algorithm.labels_[i] == cluster]
    # Find the most common true label in this cluster
    most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
    # Map this cluster to this label
    label_mapping[cluster] = most_common_label

# Use this mapping to predict labels based on cluster assignments
predicted_labels = [label_mapping[cluster] for cluster in algorithm.labels_]

# Then you can compute accuracy and other metrics using 'predicted_labels' and 'true_labels'
accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)
print(f"Accuracy: {accuracy}")

"""
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = FullyConnectedClassifier().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# Set the model to evaluation mode
model.eval()

# Variables to keep track of predictions and true labels
all_preds = []
all_true_labels = []

# Disable gradient computation since we are only predicting
with torch.no_grad():
    for images, labels in mnist_val:
        # Flatten MNIST images
        images = images.view(images.shape[0], -1).to(device)
        
        # Forward pass: compute predicted outputs
        output = model(images)
        
        # Convert output probabilities to predicted class by taking index of the max log-probability
        preds = output.argmax(dim=1, keepdim=True).squeeze()
        
        # Collect all predictions and true labels
        all_preds.append(preds.item())
        #all_preds.extend(preds.tolist())
        all_true_labels.append(labels)
        #all_true_labels.extend(labels.tolist())

# Convert lists to Tensor for performance metrics calculation
all_preds = torch.tensor(all_preds)
all_true_labels = torch.tensor(all_true_labels)

# Calculate performance metrics
accuracy = (all_preds == all_true_labels).float().mean().item()

# Print the classification report
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(all_true_labels, all_preds, digits=4))
"""

# Setup the figure and axes for a side-by-side comparison
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Plot for True Labels
scatter_true = axs[0].scatter(mu_values_2d_tsne[:, 0], mu_values_2d_tsne[:, 1], c=true_labels, cmap='tab10', alpha=0.6)
axs[0].set_title('t-SNE with True Labels')
axs[0].set_xlabel('t-SNE Dimension 1')
axs[0].set_ylabel('t-SNE Dimension 2')

# Creating a color bar for the true labels plot
colorbar_true = fig.colorbar(scatter_true, ax=axs[0], ticks=range(10))
colorbar_true.set_label('True Digit Label', rotation=270, labelpad=15)

# Plot for Predicted Labels from Clustering
scatter_pred = axs[1].scatter(mu_values_2d_tsne[:, 0], mu_values_2d_tsne[:, 1], c=algorithm.labels_, cmap='tab10', alpha=0.6)
axs[1].set_title('t-SNE with Cluster Labels')
axs[1].set_xlabel('t-SNE Dimension 1')
axs[1].set_ylabel('t-SNE Dimension 2')
# Creating a color bar for the predicted labels plot
colorbar_pred = fig.colorbar(scatter_pred, ax=axs[1], ticks=range(10))
colorbar_pred.set_label('Cluster Label', rotation=270, labelpad=15)

plt.show()