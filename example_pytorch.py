import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torchvision.models import vgg16
from torchvision.transforms import transforms
import torchvision
import foolbox
from foolbox.models import PyTorchModel
from foolbox.criteria import TargetClass

from load_cifar10_data import get_data, get_test_data
from utils import plot_result


class CustomTensorDataset(TensorDataset):
    """Dataset wrapping tensors.
    Extended to support image transformations
    """

    def __init__(self, *tensors, transform):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        pictures = self.transform(self.tensors[0][index])
        labels = self.tensors[1][index]
        return pictures, labels

    def __len__(self):
        return self.tensors[0].size(0)


'''Reference blog post https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce'''
# pick a trained model from pytorch
model = vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# change the classifier, map the internal 4096 values to 10 output classes, add batchnorm
model.classifier[3] = nn.BatchNorm1d(4096)
model.classifier[4] = nn.Linear(4096, 4096)
model.classifier[5] = nn.ReLU(True)
model.classifier[6] = nn.Sequential(nn.Dropout(),nn.BatchNorm1d(4096), nn.Linear(4096, 10))

# Find total parameters and trainable parameters(most paramaters are not trainable, which speed up training a lot
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Loss and optimizer
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# dynamically reduce lr if loss not improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5000, verbose=True)

# train the model
data = get_data()
test_data = get_test_data()
X_train = data[b'data']
X_test = test_data[b'data']
# transpose for pytorch
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
y_train = data[b'labels']
y_test = test_data[b'labels']

# normalize data(this is a very important step, which increases accuracy a lot)
# https://github.com/facebook/fb.resnet.torch/issues/180
normalize = torchvision.transforms.Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                             std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize
])
# create data and dataloder
data = CustomTensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), transform=transform)
test_data = CustomTensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(), transform=transform)
dataloader = DataLoader(data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

if os.path.exists('model.pt'):
    print('Loading model')
    model = torch.load('model.pt')

else:
    print('Training model')
    # set model for training
    model.train()
    for epoch in range(100):
        t_loss = 0.0
        total_train_images = 0.0
        total_train_correct = 0.0
        total_test_images = 0.0
        total_test_correct = 0.0

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
            scheduler.step(loss)

            t_loss += loss.item()
            total_train_correct += right_count
            total_train_images += data.shape[0]

        # eval using test set
        model.eval()
        for data, targets in test_dataloader:
            # Generate predictions
            out = model(data)
            labels = torch.argmax(targets, dim=1)
            right_count = float(torch.sum(torch.argmax(out, dim=1) == labels))
            total_test_correct += right_count
            total_test_images += data.shape[0]

        model.train()

        print(
            f'Training Loss: {t_loss}, Training Accuracy: {total_train_correct / total_train_images}, Test Accuracy: {total_test_correct / total_test_images}')

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
