import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import json
import os
import random
from torchvision.transforms import v2
from pytorch_lightning.utilities.model_summary import summarize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE


# Load json file containing variables
path = os.getcwd()
with open(os.path.join(path) + "/ProofOfConcept/parameters.json") as file:
    data = json.load(file)
    max_epochs = data['model']['max_epochs']
    batch_size = data['model']['batch_size']
    lr = data['optimizer']['learning_rate']
    weight_decay = data['optimizer']['weight_decay']
    step_size = data['optimizer']['scheduler']['step_size']
    gamma = data['optimizer']['scheduler']['gamma']
    save = data['save']

# Check if GPU is available
if torch.cuda.is_available(): 
    device = 'cuda' 
else: 
    device = 'cpu'


# Define a custom dataset class to add varying levels of noise to MNIST images
class LabeledNoisyMNIST(Dataset):
    def __init__(self, dataset, noise_levels):
        self.dataset = dataset
        #self.noise_levels = noise_levels if noise_levels else [0.0] * len(self.dataset)
        self.noise_levels = noise_levels
        self.labels = [label for _, label in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Get the noise level for this image
        noise_level = self.noise_levels[index]
        #noise_level = self.noise_levels[0][index]
        #noise_label = self.noise_levels[1]

        # Add Gaussian noise to the image with varying levels
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = torch.clamp(noisy_image, 0, 1)  # Ensure pixel values are between 0 and 1

        return noisy_image, label, noise_level#, noise_label

# PyTorch Lightning DataModule
class LabeledNoisyMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.batch_size = batch_size
        # Transforms images to a PyTorch Tensor
        self.tensor_transform = transforms.ToTensor()
        #self.tensor_transform = transforms.Compose(
        #    [
        #        transforms.ToTensor(),
        #        v2.RandomAffine(degrees=180,translate=(0.4, 0.4),scale=(0.5,0.5)),
        #    ]
        #)

        dataset = datasets.MNIST(root='./data', train=True, transform=self.tensor_transform)
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
            
    def train_dataloader(self):
        # Define varying levels of noise for each part of the dataset
        num_total_images = len(self.mnist_train)
        num_images_per_part = num_total_images // 4
        noise_levels_part1 = [0.0] * num_images_per_part
        #noise_levels_part1 = np.linspace(0.1, 0.3, num_images_per_part)
        noise_levels_part2 = [0.0] * num_images_per_part
        #noise_levels_part2 = np.linspace(0.1, 0.3, num_images_per_part)
        noise_levels_part3 = [0.0] * num_images_per_part
        #noise_levels_part3 = np.linspace(0.1, 0.3, num_images_per_part)
        noise_levels_part4 = [0.0] * num_images_per_part
        #noise_levels_part4 = np.linspace(0.1, 0.3, num_images_per_part)
        #np.random.shuffle(noise_levels_part4)

        # Combine noise levels for the entire dataset
        self.all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

        # Shuffle the noise levels to ensure random assignment to images
        np.random.shuffle(self.all_noise_levels)

        self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=self.all_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=False)
    
    def val_dataloader(self):
        self.persistent_workers=True
        # Define varying levels of noise for each part of the dataset
        num_total_images = len(self.mnist_val)
        num_images_per_part = num_total_images // 4
        noise_levels_part1 = [0.0] * num_images_per_part
        #noise_levels_part1 = np.linspace(0.5, 0.7, num_images_per_part)
        noise_levels_part2 = [0.0] * num_images_per_part
        #noise_levels_part2 = np.linspace(0.5, 0.7, num_images_per_part)
        noise_levels_part3 = [0.0] * num_images_per_part
        #noise_levels_part3 = np.linspace(0.5, 0.7, num_images_per_part)
        noise_levels_part4 = [0.0] * num_images_per_part
        #noise_levels_part4 = np.linspace(0.5, 0.7, num_images_per_part)
        #np.random.shuffle(noise_levels_part4)

        # Combine noise levels for the entire dataset
        self.all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

        # Shuffle the noise levels to ensure random assignment to images
        np.random.shuffle(self.all_noise_levels)

        self.labeled_noisy_mnist_val = LabeledNoisyMNIST(self.mnist_val, noise_levels=self.all_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=False)


# PyTorch Lightning Module with Clustering and Classification
class LabeledNoisyMNISTModelWithClusteringAndClassification(pl.LightningModule):
    def __init__(self, train_losses=[]):
        super().__init__()
        self.inputs = []
        self.outputs = []
        self.train_losses = []
        self.val_losses = []
        self.adjusted_rand_index = []
        self.adjusted_rand_index_epoch = []
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = 256
        self.latent_size = 50
        self.num_clusters = 10  # Number of clusters for K-means

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.latent_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_size, (self.latent_size * 2))# + self.num_clusters))  # Output for clustering
        )


        # Classifier head
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.num_clusters),  # Number of clusters as classifier output
            torch.nn.Softmax(dim=1)
        )
        
        """
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_clusters),
            torch.nn.Softmax(dim=1)
        )
        """

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.input_size),
            torch.nn.Sigmoid()  # Output values between 0 and 1
        )
        
        # Validation using MSE Loss function
        #self.loss_function = torch.nn.MSELoss()

    def loss_function(self, recon_x, x, mu, log_var):
        reconstruction_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence regularization
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_size], encoded[:, self.latent_size:]
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

    def training_step(self, batch, batch_idx):
        image, label, noise_level = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28 * 28)).to(device)

        # Forward pass through the encoder
        #encoded = self(image)
        recon_x, mu, log_var = self(image)

        # Check if the number of samples is greater than or equal to the number of clusters
        if image.size(0) >= self.num_clusters:
            # K-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            cluster_assignments = kmeans.fit_predict(mu.detach().cpu().numpy())

            # Assign cluster labels to classes
            cluster_labels = self.classifier_head(mu).argmax(dim=1).type(torch.FloatTensor).to(device)

            # Calculate clustering performance
            adjusted_rand_index = adjusted_rand_score(label.cpu().numpy(), cluster_assignments)

            # Log clustering performance
            self.log('adjusted_rand_index', adjusted_rand_index, prog_bar=True)

            # Classification loss
            classification_loss = torch.nn.functional.cross_entropy(cluster_labels, label.type(torch.FloatTensor).to(device))

            # VAE loss
            vae_loss = torch.nn.functional.binary_cross_entropy(recon_x, image, reduction='sum')

            # Total loss
            loss = 1 * classification_loss + vae_loss

        else:
            # If the number of samples is less than the number of clusters, set loss to 0
            loss = torch.tensor(0.0).to(device)

        self.train_losses.append(loss.item())
        self.adjusted_rand_index.append(adjusted_rand_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label, noise_level = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28 * 28)).to(device)

        # Forward pass through the encoder
        #encoded = self(image)
        recon_x, mu, log_var = self(image)

        # Check if the number of samples is greater than or equal to the number of clusters
        if image.size(0) >= self.num_clusters:
            # K-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
            cluster_assignments = kmeans.fit_predict(mu.detach().cpu().numpy())

            # Assign cluster labels to classes
            cluster_labels = self.classifier_head(mu).argmax(dim=1).type(torch.FloatTensor).to(device)

            matches = [i for i, j in zip(label, cluster_labels) if i == j]
            classification_score = len(matches)/len(label)
            # Print statements to check cluster_labels and true labels
            print("Cluster Labels (Softmax Output):", cluster_labels)
            print("True Labels:", label)
            print("Classification SCore:", classification_score)

            # Calculate clustering performance
            adjusted_rand_index = adjusted_rand_score(label.cpu().numpy(), cluster_assignments)

            # Log clustering performance
            self.log('adjusted_rand_index_epoch', adjusted_rand_index, prog_bar=True)

            # Classification loss
            classification_loss = torch.nn.functional.cross_entropy(cluster_labels, label.type(torch.FloatTensor).to(device))

            # VAE loss
            vae_loss = torch.nn.functional.binary_cross_entropy(recon_x, image, reduction='sum')

            # Total loss (sum of classification loss and VAE loss)
            loss = 1 * classification_loss + vae_loss

        else:
            # If the number of samples is less than the number of clusters, set loss to 0
            loss = torch.tensor(0.0).to(device)

        #self.train_losses.append(loss.item())
        self.log('val_loss', loss)
        self.val_losses.append(loss.item())
        self.adjusted_rand_index_epoch.append(adjusted_rand_index)
        self.inputs.append(image)
        self.outputs.append(recon_x)
        return loss
    
    def get_latent_space(self, dataloader):
        self.eval()
        latent_space = []

        for images, _, _ in dataloader:
            images = torch.reshape(images, (-1, 28*28))#.to(device)
            _, mu, _ = self(images)
            latent_space.append(mu.cpu().detach().numpy())

        return np.concatenate(latent_space, axis=0)

    def configure_optimizers(self):
        # Using an Adam Optimizer for both clustering and classification
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1, verbose=False)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }


if __name__ == '__main__':
    # Model Initialization
    model = LabeledNoisyMNISTModelWithClusteringAndClassification()
    module = LabeledNoisyMNISTDataModule()
    train_dataloaders = module.train_dataloader()
    val_dataloaders = module.val_dataloader()

    checkpoint_callback = ModelCheckpoint(
        dirpath="D:/Thesis/ProofOfConcept/saved_models/",
        filename="AutoencoderMNIST-VAE-ZERO-PosRotVar2-{epoch:02d}",
        save_on_train_epoch_end=save
    )

    trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], accelerator=device, devices=1)
    train = trainer.fit(model, module)

    # Visualize some results
    with torch.no_grad():
        images = model.inputs[-10:] # Final 10 inputs and outputs
        reconstructed_images = model.outputs[-10:]

        # Plot the original and reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=len(images), figsize=(15, 3))

        for i in range(len(images)):
            axes[0, i].imshow(images[i][0].reshape(28, 28).cpu().detach().numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed_images[i][0].reshape(28, 28).cpu().detach().numpy(), cmap='gray')
            axes[1, i].axis('off')


        # Visualize the latent space
        # Choose a batch of images from the validation set
        sample_batch = next(iter(val_dataloaders))
        images, labels, _ = sample_batch

        # Reshape images for the VAE
        images = torch.reshape(images, (-1, 28*28))

        # Forward pass through the VAE to obtain latent representations
        #mu, log_var = model.encoder(images)
        recon_x, mu, log_var = model(images)
        latent_representations = model.reparameterize(mu, log_var).cpu().numpy()

        # Scatter plot in 2D (assuming latent space is 2-dimensional)
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=labels.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Scatter Plot of Latent Space')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')


        # t-SNE
        # Extract latent space representations for the validation set
        latent_space_val = model.get_latent_space(val_dataloaders)

        # Access the original labels from the LabeledNoisyMNIST dataset
        original_labels_val = np.array(val_dataloaders.dataset.labels)

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        latent_space_tsne = tsne.fit_transform(latent_space_val)

        # Visualize t-SNE plot
        plt.figure(figsize=(10, 8))
        for i in range(10):  # 10 classes
            indices = np.where(original_labels_val == i)[0]
            #if len(indices) > 0:
            plt.scatter(latent_space_tsne[indices, 0], latent_space_tsne[indices, 1], label=f'Class {i}')

        plt.title('t-SNE Visualization of Latent Space')
        plt.legend()

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(model.train_losses, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()

        # Plot the validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(model.val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Validation Losses')
        plt.legend()

        # Plot the adjusted Rand index during training
        plt.figure(figsize=(15, 5))
        plt.plot(model.adjusted_rand_index, label='Rand Index (Training)', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Rand Index')
        plt.legend()
        plt.title('Rand Index During Training')

        # Plot the adjusted Rand index during validation
        plt.figure(figsize=(15, 5))
        plt.plot(model.adjusted_rand_index_epoch, label='Rand Index (Validation)', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Rand Index')
        plt.legend()
        plt.title('Rand Index During Validation')

        plt.show()