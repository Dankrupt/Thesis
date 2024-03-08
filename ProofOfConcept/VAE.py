import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
    subset_size = data['subset_size']

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
        #noise_level = self.noise_levels[index][0]
        #print(self.noise_levels[0])
        noise_label = self.noise_levels[index][1]
        percent = self.noise_levels[index][0]

        #noise_label = self.noise_levels[1]
        #percent = self.noise_levels[0]
        
        noisy_image = self.add_noise(image, percent)

        #print(f'noise_label: {noise_label}')
        #print(f'Percent: {percent}')
        return noisy_image, label, noise_label, percent
    
    def add_noise(self, img, percent):
        dev = img.device
        #percent = .5 # Try changing from 0 to 1
        beta = torch.tensor(percent, device=dev)
        alpha = torch.tensor(1 - percent, device=dev)
        noise = torch.randn_like(img)
        return (alpha * img + beta * noise)

# PyTorch Lightning DataModule
class LabeledNoisyMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.tensor_transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(root='./data', train=True, transform=self.tensor_transform)
        self.mnist_train, self.mnist_val = random_split(self.dataset, [55000, 5000])
            
    def train_dataloader(self):
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
        self.all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

        # Shuffle the noise levels to ensure random assignment to images
        np.random.shuffle(self.all_noise_levels)

        self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=self.all_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    def train_dataloader_blend(self):
        # Define varying levels of noise for each part of the dataset
        num_total_images = len(self.mnist_train)

        # Beta noise level
        self.percent = np.linspace(0, 0, num_total_images)
        self.labeled_noise_levels = []
        for level in self.percent:
            if 0 <= level < 0.25:
                self.labeled_noise_levels.append((level, 0))
            elif 0.25 <= level < 0.5:
                self.labeled_noise_levels.append((level, 0))
            elif 0.5 <= level < 0.75:
                self.labeled_noise_levels.append((level, 1))
            elif 0.75 <= level <= 1:
                self.labeled_noise_levels.append((level, 1))

        #print(self.labeled_noise_levels)

        self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=self.labeled_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    def val_dataloader_blend(self):
        # Define varying levels of noise for each part of the dataset
        num_total_images = len(self.mnist_train)

        # Beta noise level
        self.percent = np.linspace(0, 0, num_total_images)
        self.labeled_noise_levels = []
        for level in self.percent:
            if 0 <= level < 0.25:
                self.labeled_noise_levels.append((level, 0))
            elif 0.25 <= level < 0.5:
                self.labeled_noise_levels.append((level, 1))
            elif 0.5 <= level < 0.75:
                self.labeled_noise_levels.append((level, 2))
            elif 0.75 <= level <= 1:
                self.labeled_noise_levels.append((level, 3))

        self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=self.labeled_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    def subset_dataloader_blend(self):
        # Subset selection
        subset_indices = torch.randperm(len(self.dataset))[:subset_size]
        subset_dataset = Subset(self.dataset, subset_indices)

        # Define varying levels of noise for each part of the dataset
        num_total_images = len(subset_dataset)

        # Beta noise level
        #self.percent = np.linspace(0, 1, num_total_images)
        self.percent0 = np.linspace(0.0, 0.1, num_total_images//4)
        self.percent1 = np.linspace(0.3, 0.4, num_total_images//4)
        self.percent2 = np.linspace(0.6, 0.7, num_total_images//4)
        self.percent3 = np.linspace(0.9, 1.0, num_total_images//4)
        self.percent = np.concatenate([self.percent0, self.percent1, self.percent2, self.percent3])

        #self.percent = np.linspace(0, 1, num_total_images)

        self.labeled_noise_levels = []
        for level in self.percent:
            if 0 <= level < 0.25:
                self.labeled_noise_levels.append((level, 0))
            elif 0.25 <= level < 0.5:
                self.labeled_noise_levels.append((level, 1))
            elif 0.5 <= level < 0.75:
                self.labeled_noise_levels.append((level, 2))
            elif 0.75 <= level <= 1:
                self.labeled_noise_levels.append((level, 3))

        self.labeled_noisy_mnist_train = LabeledNoisyMNIST(subset_dataset, noise_levels=self.labeled_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    def subset_dataloader(self):        
        # Subset selection
        subset_indices = torch.randperm(len(self.dataset))[:subset_size]
        subset_dataset = Subset(self.dataset, subset_indices)

        num_total_images = len(subset_dataset)
        num_images_per_part = num_total_images // 4
        #noise_levels_part1 = [0.0] * num_images_per_part
        noise_levels_part1 = (np.linspace(0.0, 0.1, num_images_per_part), 0)
        #noise_levels_part2 = [0.0] * num_images_per_part
        noise_levels_part2 = (np.linspace(0.0, 0.1, num_images_per_part), 1)
        #noise_levels_part3 = [0.0] * num_images_per_part
        noise_levels_part3 = (np.linspace(0.89, 0.9, num_images_per_part), 2)
        #noise_levels_part4 = [0.0] * num_images_per_part
        noise_levels_part4 = (np.linspace(0.89, 0.9, num_images_per_part), 3)
        #np.random.shuffle(noise_levels_part4)

        # Combine noise levels for the entire dataset
        #self.all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])
        self.all_noise_levels = [noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4]
        self.labeled_noise_levels = []
        for i in self.all_noise_levels:
            labeled_noise_level = []
            for j in i[0]:
                labeled_noise_level.append((j, i[1]))
            self.labeled_noise_levels.extend(labeled_noise_level)

        # Shuffle the noise levels to ensure random assignment to images
        np.random.shuffle(self.labeled_noise_levels)

        self.labeled_noisy_mnist_subset = LabeledNoisyMNIST(subset_dataset, noise_levels=self.labeled_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_subset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        # Define varying levels of noise for each part of the dataset
        num_total_images = len(self.mnist_val)
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

        self.labeled_noisy_mnist_val = LabeledNoisyMNIST(self.mnist_val, noise_levels=self.all_noise_levels)
        return DataLoader(self.labeled_noisy_mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)


# PyTorch Lightning Module
class LabeledNoisyMNISTModel(pl.LightningModule):
    def __init__(self, train_losses=[]):
        super().__init__()
        self.outputs = []
        self.train_losses = []
        self.val_losses = []
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = 256
        self.latent_size = 100

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
        image, label, noise_level, percent = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)
        
        recon_x, mu, log_var = self(image)
        loss = self.loss_function(recon_x, image, mu, log_var)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, label, noise_level, percent = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)

        recon_x, mu, log_var = self(image)
        loss = self.loss_function(recon_x, image, mu, log_var)

        self.val_losses.append(loss.item())
        self.log('val_loss', loss)
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
    
    def encode(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_size], encoded[:, self.latent_size:]
        return mu, log_var


if __name__ == '__main__':
    # Model Initialization
    model = LabeledNoisyMNISTModel()
    module = LabeledNoisyMNISTDataModule()
    train_dataloaders = LabeledNoisyMNISTDataModule().train_dataloader_blend()
    val_dataloaders = LabeledNoisyMNISTDataModule().val_dataloader_blend()

    checkpoint_callback = ModelCheckpoint(
                            dirpath= "D:/Thesis/ProofOfConcept/saved_models/",
                            filename= "AutoencoderMNIST-VAE-ZERO-Blend-{epoch:02d}",
                            save_on_train_epoch_end=save)

    trainer = Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], accelerator=device, devices=1)
    trainer.fit(model,module)
    trainer.validate(model=model, datamodule=module)

    # Visualizing input and output
    image = model.outputs[-1][0][-1]
    output = model.outputs[-1][1][-1]

    # Plotting the loss values and final input and output
    with torch.no_grad():
        fig, axs = plt.subplots(1,5)
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Loss')
        axs[0].plot(model.train_losses)
        axs[0].set_title('Train Loss')
        
        axs[1].plot(model.val_losses)
        axs[1].set_title('Validation Loss')

        image = torch.reshape(image, (28, 28))
        axs[2].set_xlabel('Input')
        axs[2].imshow(image.cpu())
            
        output = torch.reshape(output, (28, 28))
        axs[3].set_xlabel('Output')
        axs[3].imshow(output.cpu())

        axs[4] = plt.hist(model.train_losses, bins=50)
        plt.show()