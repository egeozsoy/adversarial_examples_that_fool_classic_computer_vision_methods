import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torchvision.models import vgg16
import foolbox
from foolbox.models import PyTorchModel
from foolbox.criteria import TargetClass

from load_cifar10_data import get_data
from utils import plot_result

'''Reference blog post https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce'''
# pick a trained model from pytorch
model = vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Add on classifier, map the internal 4096 values to 10 output classes
model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 10))

# Find total parameters and trainable parameters(most paramaters are not trainable, which speed up training a lot
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Loss and optimizer
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# train the model
data = get_data()
# normalize data between 0 and 1(this is a very important step, which increases accuracy a lot)
X_train = data[b'data'] / 255
# transpose for pytorch
X_train = np.transpose(X_train, (0, 3, 1, 2))
y_train = data[b'labels']
# create data and dataloder
data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
dataloader = DataLoader(data, batch_size=10)

if os.path.exists('model.pt'):
    print('Loading model')
    model = torch.load('model.pt')

else:
    print('Training model')
    # set model for training
    model.train()
    for epoch in range(50):
        t_loss = 0.0
        total_images = 0.0
        total_correct = 0.0
        for data, targets in dataloader:
            # Generate predictions
            out = model(data)
            # Calculate loss
            labels = torch.argmax(targets, dim=1)
            loss = criteration(out, labels)
            right_count = float(torch.sum(torch.argmax(out, dim=1) == labels))

            loss.backward()
            # Update model parameters
            optimizer.step()

            t_loss += loss.item()
            total_correct += right_count
            total_images += data.shape[0]

        print(f'Loss : {t_loss} accuracy: {total_correct / total_images}')

    torch.save(model, 'model.pt')

# set model for evaluation(changes batch layers etc.)
model.eval()

# select an image to generate adversarial examples
image = np.float32(X_train[0:1])
# get the prediction of the model to that image as integer
label = int(np.argmax(model(torch.from_numpy(image).float()).detach().numpy()))

adversarial = None
# stop if unsuccessful after #timeout trials
timeout = 5
while adversarial is None and timeout >= 0:
    fmodel = PyTorchModel(model, bounds=(0, 255), num_classes=10)
    attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=TargetClass(1))
    # multiply the image with 255, to reverse the normalization before the activations
    adversarial = attack(image[0] * 255, label, verbose=True, iterations=100)

    timeout -= 1

print('Original image predicted as {}'.format(np.argmax(model(torch.from_numpy(image).float()).detach().numpy())))
print('Adversarial image predicted as {}'.format(np.argmax(model(torch.from_numpy(np.array([adversarial])).float()).detach().numpy())))

# before plotting, we need to change the the image shape and also reverse normalization.
plot_result(np.transpose(image, (0, 2, 3, 1)) * 255, np.transpose(adversarial, (1, 2, 0)))
