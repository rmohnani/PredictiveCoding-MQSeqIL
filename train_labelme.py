import BP
import IL_Flex
import IL_ConvBig
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
import time as timer
from LabelMeDataset import LabelMeDataset
from torch import mps

bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
softmax = torch.nn.Softmax(dim=1)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
relu = torch.nn.ReLU()

num_classes = 12

# Load Data
def get_data(batch_size=100):
    traindir = 'data/LabelMe_12_50k/train'
    testdir = 'data/LabelMe_12_50k/test'
    train_annot_bin = "data/LabelMe_12_50k/train/annotation.bin"
    train_annot_text = "data/LabelMe_12_50k/train/annotation.txt"
    test_annot_bin = "data/LabelMe_12_50k/test/annotation.bin"
    test_annot_text = "data/LabelMe_12_50k/test/annotation.txt"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = LabelMeDataset(train_annot_bin, traindir, 0, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = LabelMeDataset(test_annot_bin, testdir, 0, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)
    print(len(train_loader), len(test_loader))
    return train_loader, test_loader


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
        # print("here")
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.to(dev)
            y = y.to(dev)
            # print("there")
            target = F.one_hot(y, num_classes=num_classes).to(dev)
            target = target.to(torch.float32)
            # print("after")

            # Test and record losses and accuracy over whole test set
            
            h = model.initialize_values(images)
            # print("before")
            # print(type(h), len(h), h[-1])
            # print(softmax(h[-1]).dtype, target.dtype)
            # print("a")
            mse_val = mse(softmax(h[-1]), target)
            # print(type(mse_val))
            global_loss = torch.mean(mse_val.sum(1), dtype=torch.float32)
            # print("now")
            test_accuracies[lr][seed][-1] += utilities.compute_num_correct(softmax(h[-1]), y)
            test_losses[lr][seed][-1] += global_loss.item()
            testn += images.size(0)
            # print("over")

        test_accuracies[lr][seed][-1] /= testn
        test_losses[lr][seed][-1] /= testn


def train_model(train_loader, test_loader, model, seed, lr, test_losses, test_accuracies, epochs, dev, b_size):
    test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)

    for ep in range(epochs):
        print("ep", ep)
        for batch_idx, (images, y) in enumerate(train_loader):
            print("batch", batch_idx)
            images = images.to(dev)
            y = y.to(dev)
            target = F.one_hot(y, num_classes=num_classes)
            if images.size(0) == b_size:
                _, _ = model.train_wts(images.detach(), target.detach(), y)
            # break

        test(test_losses, test_accuracies, model, test_loader, seed, lr, dev)
        print(ep+1, 'Acc:', test_accuracies[lr][seed][-1] * 100)


def train(models, batch_size, dev, epochs, test_losses, test_accuracies):

    for l in range(len(models)):
        print(f'Training Alpha:{models[l][0].alpha}')
        for m in range(len(models[0])):

            train_loader, test_loader = get_data(batch_size)
            print("got data")
            train_model(train_loader, test_loader, models[l][m], m, l, test_losses, test_accuracies, epochs, dev, batch_size)
            print("finished training")

            print(f'Seed:{m}', f'MaxAcc:{max(test_accuracies[l][m])}',
                  f'LastAcc:{test_accuracies[l][m][-1]}')


#
def training_run(epochs=50, batch_size=64, num_seeds=1, alpha=[1], model_type=0, beta=100, arch=0, gamma=.05,
                 small = True):
    # dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # mps.empty_cache()

    models = []

    for l in range(len(alpha)):
        #Add list of seeds at this learning rate
        models.append([])
        for m in range(num_seeds):
            # BP-SGD
            if model_type == 0:
                models[-1].append(IL_Flex.IL(type=0, alpha=alpha[l], small = small, output_dim=num_classes))

            # BP-Adam
            elif model_type == 1:
                models[-1].append(IL_Flex.IL(type=1, alpha=alpha[l], small = small, output_dim=num_classes))

            # SeqIL
            elif model_type == 2:
                models[-1].append(IL_Flex.IL(n_iter=4, gamma=gamma, beta=beta, type=2, alpha=alpha[l], small = small, output_dim=num_classes))

            # SeqIL-MQ
            elif model_type == 3:
                models[-1].append(IL_Flex.IL(n_iter=4, gamma=gamma, beta=beta, type=3, alpha=alpha[l], small = small, output_dim=num_classes))

            # SeqIL-Adam
            elif model_type == 4:
                models[-1].append(IL_Flex.IL(n_iter=4, gamma=gamma, beta=beta, type=4, alpha=alpha[l], small = small, output_dim=num_classes))


        # To Device
        for i in range(len(models[-1])):
            models[-1][i].to(dev)

    #################################################
    # Create Containers

    test_losses = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]
    test_accs = [[[] for m in range(num_seeds)] for m in range(len(models))]  # [model_lr][model_seed]

    #################################################
    # Train
    print(f'\nTRAINING LABELME50K MODEL TYPE {model_type} SIZE SMALL {small}')
    train(models, batch_size, dev, epochs, test_losses, test_accs)

    # Store Data
    best_test_acc = torch.mean(torch.tensor([max(test_accs[0][x]) for x in range(len(test_accs[0]))])).item()
    best_lr = 0
    for l in range(1, len(models)):
        ac = torch.mean(torch.tensor([max(test_accs[l][x]) for x in range(len(test_accs[0]))])).item()
        if best_test_acc < ac:
            best_test_acc = ac
            best_lr = l

    with open(f'data/LabelMe50K{model_type}_small{small}_epochs{epochs}_mbatch{batch_size}_arch{arch}.data','wb') as filehandle:
        pickle.dump([test_accs[best_lr], test_losses[best_lr], alpha[best_lr]], filehandle)
