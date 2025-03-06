import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from torchvision.datasets import STL10
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
backbone = torch.nn.Sequential(*list(model.children())[:-1])

ssl_state_dict = torch.load('./output/models/simclr_model2.pth', map_location=device)

# Load the new state dict
backbone.load_state_dict(ssl_state_dict, strict=False)
backbone.to(device)

# remove the final fully connected layer ( keep the feature extractor)
#model.fc = torch.nn.Identity()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# Load the STL10 dataset
dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract features
features = []

with torch.no_grad(): # Disable gradient tracking
    backbone.eval() # Set the model to evaluation mode
    for images, _ in dataloader: # Iterate over the dataloader
        images = images.to(device) # Move images to the device
        outputs = backbone(images) # Forward pass
        outputs = outputs.view(outputs.size(0), -1)  # Flatten the outputs
        features.append(outputs)

# Concatenate the features into a single tensor
X = torch.cat(features, dim=0)
print("Features shape:", X.shape)

# Convert the tensor to a numpy array
x_np = X.detach().numpy() 

""" 
Now we are going to perform PCA, t-SNE, and K-means clustering on the extracted features.
The main goal is to visualize the high-dimensional features in a lower-dimensional space
and see if there are any clusters or patterns in the data.
"""


# Perform PCA with 2 components (for visualization)
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(x_np)

# Plot the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', s=10) # Scatter plot of the PCA components 
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of SimCLR Features')
#plt.show()
# Save the PCA plot
plt.savefig("./output/logs/pca_plot.png")
print("PCA plot saved successfully!")

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(x_np)

# Plot the t-SNE results
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title("t-SNE of Feature Extractor Output")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
#plt.show()
# Save the t-SNE plot
plt.savefig("./output/logs/tsne_plot.png")
print("t-SNE plot saved successfully!")

# Perform K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(x_np)

# Plot the K-means clustering results
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title("t-SNE with KMeans Clustering")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
#plt.show()

# Save the K-means clustering plot
plt.savefig("./output/logs/kmeans_plot.png")
print("K-means plot saved successfully!")

