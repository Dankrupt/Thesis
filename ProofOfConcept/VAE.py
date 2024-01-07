import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import matplotlib
import json
import os


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

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()


# Define a custom dataset class to add varying levels of noise to MNIST images
class LabeledNoisyMNIST(Dataset):
    def __init__(self, dataset, noise_levels):
        self.dataset = dataset
        #self.noise_levels = noise_levels if noise_levels else [0.0] * len(self.dataset)
        self.noise_levels = noise_levels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Get the noise level for this image
        noise_level = self.noise_levels[index]

        # Add Gaussian noise to the image with varying levels
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = torch.clamp(noisy_image, 0, 1)  # Ensure pixel values are between 0 and 1

        return noisy_image, label, noise_level

# PyTorch Lightning DataModule
class LabeledNoisyMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.tensor_transform = transforms.ToTensor()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = datasets.MNIST(root='./data', train=True, transform=self.tensor_transform)
            self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

            # Define varying levels of noise for each part of the dataset
            num_total_images = len(self.mnist_train)
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
            all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

            # Shuffle the noise levels to ensure random assignment to images
            np.random.shuffle(all_noise_levels)

            # Create LabeledNoisyMNIST datasets
            self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=all_noise_levels)
            self.labeled_noisy_mnist_val = LabeledNoisyMNIST(self.mnist_val, noise_levels=all_noise_levels)

    def train_dataloader(self):
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.labeled_noisy_mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4)


# PyTorch Lightning Module
class LabeledNoisyMNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.losses = []
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = 256
        self.latent_size = 20

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.latent_size * 2)  # Two outputs for mean and log-variance
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
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
        return mu + eps * std
        
    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_size], encoded[:, self.latent_size:]
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

    def training_step(self, batch):
        image, label, noise_level = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)
        
        recon_x, mu, log_var = self(image)
        loss = self.loss_function(recon_x, image, mu, log_var)
        self.losses.append(loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        image, label, noise_level = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)

        recon_x, mu, log_var = self(image)
        loss = self.loss_function(recon_x, image, mu, log_var)

        self.outputs.append((image, recon_x))
        return loss

    def configure_optimizers(self):
        # Using an Adam Optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr = lr,
                                    weight_decay = weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, 
                                              gamma=gamma, 
                                              last_epoch=-1, 
                                              verbose=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": sch,
            "monitor": "test_loss",
            "interval": "epoch"
            }
        }


if __name__ == '__main__':
    # Model Initialization
    model = LabeledNoisyMNISTModel()
    module = LabeledNoisyMNISTDataModule()

    #if not save:
    #    checkpoint_callback = None
    #else:
    checkpoint_callback = ModelCheckpoint(
                            dirpath= "D:/Thesis/ProofOfConcept/saved_models/",
                            filename= "AutoencoderMNIST-VAE-HIGH-{epoch:02d}",
                            save_on_train_epoch_end=save)

    trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], accelerator=device, devices=1)
    trainer.fit(model,module)
    trainer.test(model=model, datamodule=module)


    # Visualizing input and output
    image = model.outputs[-1][0][-1]
    output = model.outputs[-1][1][-1]

    # Plotting the loss values and final input and output
    with torch.no_grad():
        fig, axs = plt.subplots(1,3)
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Loss')
        axs[0].plot(model.losses)
        
        image = torch.reshape(image, (28, 28))
        axs[1].set_xlabel('Input')
        axs[1].imshow(image.cpu())
            
        output = torch.reshape(output, (28, 28))
        axs[2].set_xlabel('Output')
        axs[2].imshow(output.cpu())
        plt.show()