import torch
from torchvision import transforms
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
from VAE import LabeledNoisyMNISTDataModule
from VAE import LabeledNoisyMNISTModel
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np
import seaborn as sns
from mord import LogisticAT
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    # Initialize the model and data
    tensor_transform = transforms.ToTensor()
    path = os.path.join(os.getcwd())
    model = LabeledNoisyMNISTModel.load_from_checkpoint(path + '/ProofOfConcept/saved_models/AutoencoderMNIST-VAE-ZERO-Blend-epoch=09-lat100.ckpt')
    datamodule = LabeledNoisyMNISTDataModule()
    dataloader = datamodule.subset_dataloader_blend()

    model.eval()  # Set the model to evaluation mode

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess and encode images
    encoded_images_subset = []
    true_labels = []
    noise_levels = []
    latent_features = []
    reconstructed = []
    for idx, batch in enumerate(dataloader):
        img, label, noise_label, percent = batch
        img = img.view(-1, 28*28).to(device)
        mu, log_var = model.encode(img)

        encoded_images_subset.append((mu, log_var))
        latent_features.append(mu)
        #true_labels.extend(noise_label.tolist())
        for label in noise_label:
            true_labels.append(label.tolist())

        for level in percent:
            noise_levels.append(level.tolist())


    # Ensure mu_values is a 2D array before clustering
    mu_values_2d = np.vstack([mu.cpu().detach().numpy() for mu, _ in encoded_images_subset])

    # Reduce dimensions of mu_values to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    mu_values_2d_tsne = tsne.fit_transform(mu_values_2d)

    # 3D t-SNE
    tsne = TSNE(n_components=3, random_state=0)
    mu_values_3d_tsne = tsne.fit_transform(mu_values_2d)  # This will now be 3D


    # Set up clustering algorithm
    kmeans = KMeans(n_clusters=4, random_state=42)
    spectral = SpectralClustering(n_clusters=4, random_state=42)
    gmm = GaussianMixture(n_components=4, random_state=42)
    miniBatchKMeans = MiniBatchKMeans(n_clusters=4, random_state=42)
    algorithm = kmeans

    # Cluster the encoded images in 2D
    cluster2D = algorithm.fit(mu_values_2d_tsne)

    # Cluster the encoded images in 3D
    cluster3D = algorithm.fit(mu_values_3d_tsne)

    # Assign cluster labels for classification
    label_mapping = {}
    for cluster in set(cluster2D.labels_):
        # Find all true labels for data points in this cluster
        labels_in_cluster = [true_labels[i] for i in range(len(true_labels)) if cluster2D.labels_[i] == cluster]
        
        # Check if there are any labels in this cluster before proceeding
        if labels_in_cluster:  # This ensures that the list is not empty
            # Find the most common true label in this cluster
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            # Map this cluster to this label
            label_mapping[cluster] = most_common_label
        else:
            # Handle the case where a cluster has no labels
            print(f"Cluster {cluster} has no labels, assigning default label 0.")
            label_mapping[cluster] = 0

    # Use this mapping to predict labels based on cluster assignments
    predicted_labels = [label_mapping[cluster] for cluster in cluster2D.labels_]

    # Convert tensors to lists of scalars if they are not already
    true_labels_scalar = [int(label) for label in true_labels]
    predicted_labels_scalar = [int(label) for label in predicted_labels]

    # Compute the accuracy
    accuracy = sum(1 for true, pred in zip(true_labels_scalar, predicted_labels_scalar) if true == pred) / len(true_labels_scalar)
    print(f"2D Clustering Accuracy: {accuracy}")


    # Assign cluster labels for classification
    label_mapping = {}
    for cluster in set(cluster3D.labels_):
        # Find all true labels for data points in this cluster
        labels_in_cluster = [true_labels[i] for i in range(len(true_labels)) if cluster3D.labels_[i] == cluster]
        
        # Check if there are any labels in this cluster before proceeding
        if labels_in_cluster:  # This ensures that the list is not empty
            # Find the most common true label in this cluster
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            # Map this cluster to this label
            label_mapping[cluster] = most_common_label
        else:
            # Handle the case where a cluster has no labels
            print(f"Cluster {cluster} has no labels, assigning default label 0.")
            label_mapping[cluster] = 0

    # Use this mapping to predict labels based on cluster assignments
    predicted_labels3D = [label_mapping[cluster] for cluster in cluster3D.labels_]

    # Convert tensors to lists of scalars if they are not already
    true_labels_scalar3D = [int(label) for label in true_labels]
    predicted_labels_scalar3D = [int(label) for label in predicted_labels3D]

    # Compute the accuracy
    accuracy = sum(1 for true, pred in zip(true_labels_scalar3D, predicted_labels_scalar3D) if true == pred) / len(true_labels_scalar3D)
    print(f"3D Clustering Accuracy: {accuracy}")


    # Ordinal regression
    # Ordinal Regression t-SNE
    X = mu_values_2d_tsne
    y = np.array(true_labels)

    ordinal_model = LogisticAT()
    ordinal_model.fit(X, y)

    predicted_labels_tsne = ordinal_model.predict(X)
    accuracy = accuracy_score(y, predicted_labels_tsne)
    print(f'Ordinal Regression Accuracy t-SNE: {accuracy}')
    
    true_labels = np.array(true_labels)

    noise_label = np.array(noise_label)
    latent_features = latent_features[0].detach().cpu().numpy()


    # Ordinal Regression without dimensionality reduction of latent space
    all_latent_features = []
    all_labels = []

    for batch in dataloader:  # Assuming data_loader is your DataLoader object
        img, _, labels, _ = batch  # Assuming each batch returns images and corresponding labels
        img = img.view(-1, 28*28).to(device)
        with torch.no_grad():
            mu, _ = model.encode(img)  # Get the latent features for the batch
            all_latent_features.append(mu.cpu().numpy())  # Convert to NumPy and store
            all_labels.append(labels.numpy())  # Store labels as NumPy array

    # Concatenate all batch data
    X_all = np.concatenate(all_latent_features, axis=0)  # This will be your complete feature matrix
    y_all = np.concatenate(all_labels, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    ordinal_model.fit(X_train, y_train)
    #y_pred = ordinal_model.predict(X_test)
    y_pred_all = ordinal_model.predict(X_all)

    tsne = TSNE(n_components=2, random_state=0)
    X_all_tsne = tsne.fit_transform(X_all)

    accuracy = accuracy_score(y_all, y_pred_all)
    print(f'Ordinal Regression Accuracy Latent Space: {accuracy}')

    f1 = f1_score(y_all, y_pred_all, labels=[0,1,2,3], average=None)
    print(f'Ordinal Regression F1 Score Latent Space: {f1}')



    # k-means latent space
    k_latent = kmeans.fit(X_all)

    # Assign cluster labels for classification
    label_mapping = {}
    for cluster in set(k_latent.labels_):
        # Find all true labels for data points in this cluster
        labels_in_cluster = [true_labels[i] for i in range(len(true_labels)) if k_latent.labels_[i] == cluster]
        
        # Check if there are any labels in this cluster before proceeding
        if labels_in_cluster:  # This ensures that the list is not empty
            # Find the most common true label in this cluster
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            # Map this cluster to this label
            label_mapping[cluster] = most_common_label
        else:
            # Handle the case where a cluster has no labels
            print(f"Cluster {cluster} has no labels, assigning default label 0.")
            label_mapping[cluster] = 0

    # Use this mapping to predict labels based on cluster assignments
    predicted_labels_k_latent = [label_mapping[cluster] for cluster in k_latent.labels_]

    # Convert tensors to lists of scalars if they are not already
    true_labels_scalar_k_latent = [int(label) for label in true_labels]
    predicted_labels_scalar_k_latent = [int(label) for label in predicted_labels_k_latent]

    # Compute the accuracy
    accuracy_k_latent = sum(1 for true, pred in zip(true_labels_scalar_k_latent, predicted_labels_scalar_k_latent) if true == pred) / len(true_labels_scalar)
    print(f"k Means Latent Space Clustering Accuracy: {accuracy_k_latent}")




    # Visualization
    # Ordinal regression 2D t-SNE
    plt.figure(figsize=(20, 8))

    # Plot for True Labels
    plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6)
    plt.title('True Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Plot for Predicted Labels
    plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels_scalar_k_latent, cmap='tab10', alpha=0.6)
    plt.title('Predicted Labels by Clustering Latent Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')


    # Ordinal Regression Latent Space 
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_all, y_pred_all, labels=[0,1,2,3], normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Ordinal Regression Latent Space')

    plt.figure(figsize=(20, 8))

    # Scatter plot for true labels
    plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
    plt.scatter(X_all_tsne[:, 0], X_all_tsne[:, 1], c=y_all, cmap='tab10', alpha=0.6)
    plt.title('True Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.clim(-0.5, len(np.unique(y_test))-0.5)

    # Scatter plot for predicted labels
    plt.subplot(1, 2, 2)  # Second subplot in the same 1x2 grid
    plt.scatter(X_all_tsne[:, 0], X_all_tsne[:, 1], c=y_pred_all, cmap='tab10', alpha=0.6)
    plt.title('Predicted Labels by Ordinal Regression')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.clim(-0.5, len(np.unique(y_test))-0.5)


    # t-SNE
    # Create a set of all possible labels (from both true and predicted)
    all_predicted_labels = set(label_mapping.values())  # Predicted labels from mapping

    # Create a mapping from unique true labels to an integer index
    unique_true_labels = sorted(set(true_labels))
    label_color_mapping = {label: idx for idx, label in enumerate(unique_true_labels)}

    # Map your true labels and predicted cluster labels to these colors
    true_label_colors = [label_color_mapping[label] for label in true_labels]
    predicted_label_colors = [label_color_mapping[label_mapping.get(cluster, -1)] for cluster in cluster3D.labels_]

    # Visualization with adjusted color mappings
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for True Labels with consistent colors across all known labels
    scatter_true = axs[0].scatter(mu_values_2d_tsne[:, 0], mu_values_2d_tsne[:, 1], c=true_label_colors, cmap='tab10', alpha=0.6)
    axs[0].set_title('t-SNE with True Labels')
    axs[0].set_xlabel('t-SNE Dimension 1')
    axs[0].set_ylabel('t-SNE Dimension 2')

    # Plot for Predicted Labels ensuring all labels have a color
    scatter_pred = axs[1].scatter(mu_values_2d_tsne[:, 0], mu_values_2d_tsne[:, 1], c=predicted_label_colors, cmap='tab10', alpha=0.6)
    axs[1].set_title('t-SNE with Cluster Labels')
    axs[1].set_xlabel('t-SNE Dimension 1')
    axs[1].set_ylabel('t-SNE Dimension 2')


    # 3D Visualization
    fig = plt.figure(figsize=(20, 8))

    # 3D scatter plot for the true labels
    ax_true = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_true = ax_true.scatter(mu_values_3d_tsne[:, 0], mu_values_3d_tsne[:, 1], mu_values_3d_tsne[:, 2], c=true_label_colors, cmap='tab10', alpha=0.6)
    ax_true.set_title('3D t-SNE with True Labels')
    ax_true.set_xlabel('t-SNE Dimension 1')
    ax_true.set_ylabel('t-SNE Dimension 2')
    ax_true.set_zlabel('t-SNE Dimension 3')

    # 3D scatter plot for the clustering results
    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')
    scatter_pred = ax_pred.scatter(mu_values_3d_tsne[:, 0], mu_values_3d_tsne[:, 1], mu_values_3d_tsne[:, 2], c=cluster3D.labels_, cmap='tab10', alpha=0.6)
    ax_pred.set_title('3D Clustering Results')
    ax_pred.set_xlabel('t-SNE Dimension 1')
    ax_pred.set_ylabel('t-SNE Dimension 2')
    ax_pred.set_zlabel('t-SNE Dimension 3')


    # Unsupervised Clustering Latent Space
    plt.figure(figsize=(20, 8))

    # Plot for True Labels
    plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6)
    plt.title('True Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    #plt.colorbar(label='Ordinal Class')

    # Plot for Predicted Labels
    plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels_tsne, cmap='tab10', alpha=0.6)
    plt.title('Predicted Labels by Ordinal Regression t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.show()