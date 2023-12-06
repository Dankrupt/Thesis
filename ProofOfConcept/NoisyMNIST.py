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
        #self.transform = transforms.Compose([
        #    transforms.ToTensor(),
        #])

    #def prepare_data(self):
    #    datasets.MNIST(root='./data', train=True, download=True)
    #    datasets.MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset = datasets.MNIST(root='./data', train=True, transform=self.tensor_transform)
            #self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
            self.mnist_train = dataset

            # Define varying levels of noise for each part of the dataset
            num_total_images = len(dataset)
            num_images_per_part = num_total_images // 4
            noise_levels_part1 = [0.0] * num_images_per_part
            noise_levels_part2 = [0.0] * num_images_per_part
            #noise_levels_part2 = np.linspace(0.1, 0.3, num_images_per_part)
            noise_levels_part3 = [0.0] * num_images_per_part
            #noise_levels_part3 = np.linspace(0.3, 0.5, num_images_per_part)
            noise_levels_part4 = np.linspace(0.5, 0.7, num_images_per_part)
            np.random.shuffle(noise_levels_part4)

            # Combine noise levels for the entire dataset
            all_noise_levels = np.concatenate([noise_levels_part1, noise_levels_part2, noise_levels_part3, noise_levels_part4])

            # Shuffle the noise levels to ensure random assignment to images
            #np.random.shuffle(all_noise_levels)

            # Create LabeledNoisyMNIST datasets
            self.labeled_noisy_mnist_train = LabeledNoisyMNIST(self.mnist_train, noise_levels=all_noise_levels)

    def train_dataloader(self):
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.labeled_noisy_mnist_train, batch_size=self.batch_size, shuffle=False, num_workers=4)


# PyTorch Lightning Module
class LabeledNoisyMNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.losses = []

        # Linear encoder
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
        
        # Linear decoder
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Tanh()
        )

        # Validation using MSE Loss function
        self.loss_function = torch.nn.MSELoss()
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch):
        image, label, noise_level = batch

        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)
        
        # Output of Autoencoder
        reconstructed = model(image)
        
        # Calculating the loss function
        loss = self.loss_function(reconstructed, image)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.losses.append(loss.item())
        #self.outputs.append((image, reconstructed))
        return loss
    
    def test_step(self, batch, batch_idx):
        image, label, noise_level = batch
        # Reshaping the image to (-1, 784)
        image = torch.reshape(image, (-1, 28*28)).to(device)
        
        # Output of Autoencoder
        reconstructed = model(image)
        
        # Calculating the loss function
        loss = self.loss_function(reconstructed, image)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.outputs.append((image, reconstructed))
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

    checkpoint_callback = ModelCheckpoint(
                            #monitor = "test_loss",#monitors val loss
                            #mode = "min",#Picks the fold with the lowest val_loss
                            dirpath= "saved_models/",
                            filename= "AutoencoderMNIST-DEL4-{epoch:02d}-{train_loss:.2f}",
                            save_on_train_epoch_end=False
    )

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