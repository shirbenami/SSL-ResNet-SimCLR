import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from model.resnet import build_resnet50
from trainers.train import train_model
from trainers.validate import validate_model
from trainers.test import test_model

def main():

    # Settings
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_weights_path = "output/models/simclr_model2.pth"  # Path to SSL weights

    # Data transformations
    transform = Compose([
        RandomResizedCrop(96),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset_full = STL10(root='./data', split='train', download=True, transform=transform)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    test_dataset = STL10(root='./data', split='test', download=True, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = [str(i) for i in range(10)]

    print(f"Train Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    # Build model
    model = build_resnet50()
    model.to(device)

    # Load SSL weights
    print("Loading SSL weights...")
    ssl_weights = torch.load(ssl_weights_path, map_location=device)
    model.load_state_dict(ssl_weights, strict=False)  # strict=False to allow for slight differences
    print("SSL weights loaded successfully.")

    # Freeze ResNet layers
    for param in model.parameters():  # Assuming model[0] is the ResNet encoder
        param.requires_grad = False


    for param in model.fc.parameters():
        param.requires_grad = True


    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)




    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
    # Training and validation loops
    top1_train_accuracies = []
    top1_test_accuracies = []
    top5_test_accuracies = []

    for epoch in range(num_epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_train_accuracies.append(top1_train_accuracy.item())

        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        top1_test_accuracies.append(top1_accuracy.item())
        top5_test_accuracies.append(top5_accuracy.item())
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")


    
    # Save results as graphs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Plot Top1 Training Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), top1_train_accuracies, label='Top1 Train Accuracy', color='blue')
    plt.plot(range(num_epochs), top1_test_accuracies, label='Top1 Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Top1 Accuracy')
    plt.legend()

    # Plot Top5 Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), top5_test_accuracies, label='Top5 Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Top5 Accuracy')
    plt.title('Top5 Test Accuracy')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the top 1 and top 5 classification model
    torch.save(model.state_dict(), "./output/models/top1_top5_model.pth")
    print("top 1 and top5 classification model saved successfully!")
    

if __name__ == '__main__':
    main()