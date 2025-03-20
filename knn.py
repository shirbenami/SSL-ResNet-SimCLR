import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import torch.nn as nn
from torchvision import models
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize


def generate_embeddings(model, dataloader):
    """
    Generates representations for all images in the dataloader with the given model
    """
    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, fnames in dataloader:
            #img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames

# Normalize function (if needed)
def normalize(embeddings):
    """Normalize the embeddings to unit length"""
    norm = torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings / norm


def get_image_as_np_array(img_tensor: str):
    """Returns an image as an numpy array"""
    #img = Image.open(filename)
    #return np.asarray(img)
    img = transforms.ToPILImage()(img_tensor)  
    return np.asarray(img)


# This function plots images and their nearest neighbors based on their embeddings.
# It uses K-Nearest Neighbors (KNN) to find the closest images by computing the Euclidean distance between their feature vectors.
def plot_knn_examples(embeddings, filenames,test_dataset, n_neighbors=7, num_examples=6):
    """
    This function selects random samples from the dataset and finds their nearest neighbors using KNN.
    It then visualizes the query image along with its nearest neighbors.

    Parameters:
    - embeddings: the feature vectors (embeddings) of the images.
    - filenames: list of image filenames corresponding to the embeddings.
    - n_neighbors: number of neighbors to display (default is 3).
    - num_examples: number of random examples to plot (default is 6).

    KNN is used to compute the Euclidean distance between the embeddings of images.
    The closer the distance, the more similar the images are.
    """
        
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            
            # get the correponding filename for the current index
            img_tensor = test_dataset[neighbor_idx][0]  
            #img_np = get_image_as_np_array(img_tensor)  
            img_np = img_tensor.permute(1, 2, 0).numpy()  # Convert tensor to numpy array (C x H x W to H x W x C)

             # plot the image
            print("plot the image")
            plt.imshow(img_np) 
           
            #fname = os.path.join(test_dataset, filenames[neighbor_idx])
            #plt.imshow(get_image_as_np_array(fname))
            
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")
            
        # Save the PCA plot
        plt.savefig(f"./output/logs/img_{neighbor_idx}.png")
            



"""
Build a ResNet50 model for the STL10 dataset with a projection head
"""

# define the backbone of the model
model = models.resnet50(pretrained=True)

backbone= model.backbone = torch.nn.Sequential(*list(model.children())[:-1])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ResNet50 model with pretrained weights
resnet = models.resnet50(pretrained=True)

num_classes = 10  # STL10 includes 10 classes

# Get the number of input features for the final fully connected layer
in_features = resnet.fc.in_features

# Replace the fully connected layer with an Identity layer
resnet.fc = nn.Identity()

# Add a projection head (MLP)
projection_head = nn.Sequential(
    nn.Linear(in_features, 128),  # Hidden dimension
    nn.ReLU(),
    nn.Linear(128, 128)  # Output dimension
)

# Combine ResNet and Projection Head
model = nn.Sequential(resnet, projection_head)
model.backbone = backbone

"""
Load the SSL model and generate embeddinggs to the model on the STL10 dataset
"""

ssl_state_dict = torch.load('./output/models/simclr_model2.pth', map_location=device)

# Load the new state dict
model.load_state_dict(ssl_state_dict, strict=False)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Data transformations
transform = Compose([
    #RandomResizedCrop(96),
    #RandomHorizontalFlip(),
    ToTensor(),
    #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the STL10 dataset
test_dataset = STL10(root='./data', split='test', download=True, transform=transform)

dataloader_test = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

# Generate embeddings for the test dataset using the model
embeddings, filenames = generate_embeddings(model, dataloader_test)

# Plot the KNN examples
plot_knn_examples(embeddings, filenames,test_dataset)

