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
def plot_knn_examples(embeddings, filenames,test_dataset, n_neighbors, num_examples=6):
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
            

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    """
    Perform KNN classification on the test set using the embeddings from the train set.
    Args:
    - train_features: embeddings of the train set
    - train_labels: labels for the train set
    - test_features: embeddings of the test set
    - test_labels: labels for the test set
    - k: number of nearest neighbors to consider
    - T: temperature scaling for softmax
    - num_classes: number of classes (default is 10 for STL10)

    Returns:
    - top1: top-1 accuracy
    - top5: top-5 accuracy
    """
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)  # one-hot encoding

    if isinstance(train_labels, torch.Tensor) is False:
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(train_features.device)
    
    if isinstance(test_labels, torch.Tensor) is False:
        test_labels = torch.tensor(test_labels, dtype=torch.long).to(test_features.device)
        
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        
        # getting the labels for the top-k neighbors
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        # create a one-hot encoding for the nearest neighbors
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        
        #  calculate the exponential of the distances
        distances_transform = distances.clone().div_(T).exp_()
        
        # calculate the probabilities of the nearest neighbors
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        
        # sort the probabilities and get the predictions 
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

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
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
test_embeddings, test_filenames = generate_embeddings(model, dataloader_test)

# Plot the KNN examples
#plot_knn_examples(test_embeddings, test_filenames,test_dataset,n_neighbors=7)

# load the train dataset
train_dataset = STL10(root='./data', split='train', download=True, transform=transform)

dataloader_train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

train_embeddings, train_filenames = generate_embeddings(model, dataloader_train)

# Perform KNN classification
top1_accuracy, top5_accuracy = knn_classifier(train_embeddings, train_dataset.labels, test_embeddings, test_dataset.labels, k=5, T=0.07, num_classes=10)

print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")