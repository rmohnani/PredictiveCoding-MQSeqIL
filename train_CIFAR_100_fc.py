import BP
import IL
import torch
import torchvision
import torch.optim as optim
import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import math
import torch.nn.functional as F

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()

# Load Data
def get_data(batch_size=64, data=5):
    
    d_name = 'CIFAR_100'
    num_train = 50000

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=5000, shuffle=False)

    return train_loader, test_loader, d_name, num_train



def compute_means(data):
    with torch.no_grad():
        d_tensor = torch.tensor(data[0]).view(1, -1)
        for m in range(1, len(data)):
            d_tensor = torch.cat((d_tensor, torch.tensor(data[m]).view(1, -1)), dim=0)
        return torch.mean(d_tensor, dim=0)



def test(test_losses, test_accuracies, model, test_loader, seed, lr, dev):
    with torch.no_grad():
        test_accuracies[lr][seed].append(0)
        test_losses[lr][seed].append(0)
        testn = 0

        # Test and record losses and accuracy over test set
        for batch_idx, (images, y) in enumerate(test_loader):
            # print(batch_idx)
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=100).to(dev)
            target = target.to(torch.float32)
            h = model.initialize_values(images)
            global_loss = torch.mean(mse(softmax(h[-1]), target).sum(1))
            test_accuracies[lr][seed][-1] += utilities.compute_num_correct(softmax(h[-1]), y)
            test_losses[lr][seed][-1] += global_loss.item()
            testn += images.size(0)

        test_accuracies[lr][seed][-1] /= testn
        test_losses[lr][seed][-1] /= testn



def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_accuracies, epochs, dev, b_size):
    #Initial test
    test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)

    #Test each epoch
    for ep in range(epochs):
        print("ep", ep)
        for batch_idx, (images, y) in enumerate(train_loader):
            # print(batch_idx)
            images = images.view(y.size(0), -1).to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=100)
            # target = target.to(torch.float32)
            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), target.detach(), y.detach())

        # Test every epoch
        test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)
        #print(ep + 1, 'Acc:', test_accuracies[lr][seed][-1]*100)



def train(models, batch_size, data, dev, epochs, test_losses, test_accuracies, b_size):

    for l in range(len(models)):
        print(f'Training Alpha:{models[l][0].alpha}')
        for m in range(len(models[0])):

            train_loader, test_loader, d_name, num_train = get_data(batch_size, data=data)
            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_accuracies, epochs, dev, b_size)

            print(f'Seed:{m}', f'MaxAcc:{max(test_accuracies[l][m])}',
                  f'MaxEarlyAcc:{max(test_accuracies[l][m][0:15])}',
                  f'LastAcc:{test_accuracies[l][m][-1]}')




def training_run(epochs=50, batch_size=64, data=5, num_seeds=1, alpha=[1], beta=100, model_type=0, n_hlayers=3):

    # Create Models
    model_dim = [3072]
    for ln in range(n_hlayers):
        model_dim.append(1024)
    model_dim.append(100)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    models = []

    for l in range(len(alpha)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):
            # BP-SGD
            if model_type == 0:
                models[-1].append(BP.BP(model_dim, type=0, alpha=alpha[l], smax=True))

            # BP-Adam
            elif model_type == 1:
                models[-1].append(BP.BP(model_dim, type=1, alpha=alpha[l], smax=True))

            # SeqIL
            elif model_type == 2:
                models[-1].append(IL.IL(model_dim, smax=True, n_iter=3, gamma=.05, beta=beta, type=0, alpha=alpha[l]))

            # SeqIL-MQ
            elif model_type == 3:
                models[-1].append(IL.IL(model_dim, smax=True, n_iter=3, gamma=.05,  beta=beta, type=1, alpha=alpha[l]))

            # SeqIL-Adam
            elif model_type == 4:
                models[-1].append(IL.IL(model_dim, smax=True, n_iter=3, gamma=.05, beta=beta, type=2, alpha=alpha[l]))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers
    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_accs = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################
    # Train
    print(f'\nTRAINING CIFAR100 MODEL TYPE {model_type} HIDDEN {n_hlayers}')
    train(models, batch_size, data, dev, epochs, test_losses, test_accs, b_size=batch_size)

    # Store Data
    best_test_acc = torch.mean(torch.tensor([max(test_accs[0][x]) for x in range(len(test_accs[0]))])).item()
    best_lr = 0
    for l in range(1, len(models)):
        ac = torch.mean(torch.tensor([max(test_accs[l][x]) for x in range(len(test_accs[0]))])).item()
        if best_test_acc < ac:
            best_test_acc = ac
            best_lr = l


    #print(f'Best Learning Rate, at Iterations{max_iters}, Model Type{model_type}:', best_lr)
    with open(f'data/MLP_Type{model_type}_data{data}_hlayers{n_hlayers}_epochs{epochs}_{batch_size}.data','wb') as filehandle:
        pickle.dump([test_accs[best_lr], test_losses[best_lr], alpha[best_lr]], filehandle)
