import BP
import IL
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import math
import torch.nn.functional as F
import time as timer

# Function to load data for weight analysis
def get_data(batch_size=64, data=0):
    """
    Load data for weight analysis.

    Args:
    - batch_size (int): Batch size for data loading
    - data (int): Data option (0 for SVHN, 1 for CIFAR)

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data
    - test_loader (torch.utils.data.DataLoader): DataLoader for testing data
    - d_name (str): Name of the dataset
    - num_train (int): Number of training samples
    """
    if data == 0:
        d_name = 'SVHN'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.SVHN(root='./data', download=True, split='train', transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', download=True, split='test', transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)

    elif data == 1:
        d_name = 'CIFAR'
        num_train = 50000

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                                  shuffle=False)

    return train_loader, test_loader, d_name, num_train


# Function to train the model for weight analysis
def train_model(train_loader, model, epochs, dev):
    """
    Train the model for weight analysis.

    Args:
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data
    - model (torch.nn.Module): Neural network model
    - epochs (int): Number of epochs for training
    - dev (str): Device to run training on (e.g., 'cuda' or 'cpu')
    """
    for ep in range(epochs):
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=10)
            _, _ = model.train_wts(images.detach(), target.detach(), y)


# Function to orchestrate model training for weight analysis
def train(model, batch_size, data, dev, epochs):
    """
    Orchestrates model training for weight analysis.

    Args:
    - model (torch.nn.Module): Neural network model
    - batch_size (int): Batch size for data loading
    - data (int): Data option (0 for SVHN, 1 for CIFAR)
    - dev (str): Device to run training on (e.g., 'cuda' or 'cpu')
    - epochs (int): Number of epochs for training

    Returns:
    - float: Training time
    """
    train_loader, _, _, _ = get_data(batch_size, data=data)
    start = timer.perf_counter()
    train_model(train_loader, model, epochs, dev)
    end = timer.perf_counter()
    return end - start


# Function to run the weight analysis training
def training_run(epochs=10, batch_size=64, data=1, model_type=0, n_hlayers=3):
    """
    Run weight analysis training.

    Args:
    - epochs (int): Number of epochs for training
    - batch_size (int): Batch size for data loading
    - data (int): Data option (0 for SVHN, 1 for CIFAR)
    - dev (str): Device to run training on (e.g., 'cuda' or 'cpu')
    - model_type (int): Type of neural network model
    - n_hlayers (int): Number of hidden layers in the model
    """
    # Create Model
    model_dim = [3072]
    for ln in range(n_hlayers):
        model_dim.append(1024)
    model_dim.append(10)

    # Create the model based on the model type
    if model_type == 0:
        model = BP.BP(model_dim, type=0, alpha=.01, smax=True)
    elif model_type == 1:
        model = BP.BP(model_dim, type=1, alpha=.000045, smax=True)
    elif model_type == 2:
        model = IL.IL(model_dim, smax=True, n_iter=3, gamma=.05, alpha=1.5, type=0, beta=100)
    elif model_type == 3:
        model = IL.IL(model_dim, smax=True, n_iter=3, gamma=.05,  alpha=.000015, type=2, beta=100)
    elif model_type == 4:
        model = IL.IL(model_dim, smax=True, n_iter=3, gamma=.05, alpha=.00005, type=3, beta=100)
    elif model_type == 5:
        model = IL.IL(model_dim, smax=True, n_iter=15, gamma=.05, alpha=1.5, type=0, beta=.1)
    elif model_type == 6:
        model = IL.IL(model_dim, smax=True, n_iter=15, gamma=.05, alpha=.000015, type=2, beta=100)
    elif model_type == 7:
        model = IL.IL(model_dim, smax=True, n_iter=15, gamma=.05, alpha=.00005, type=3, beta=100)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Move model to device
    model.to(dev)
    # print('here')
    # Train the model and measure time
    time = train(model, batch_size, data, dev, epochs)
    print(f'Fully Connected, Model Type:{model_type}  Time:{time}')

    # Save weight data
    with open(f'data/Time_Type{model_type}_data{data}_epochs{epochs}.data','wb') as filehandle:
        pickle.dump(time, filehandle)
