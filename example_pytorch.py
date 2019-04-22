import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torchvision.models import resnet18
from torchvision.transforms import transforms
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
use_gpu = torch.cuda.is_available()

# pick a trained model from pytorch
model = resnet18(pretrained=True)

# We can choose to freeze model weights, by setting required grad to False
for param in model.parameters():
    param.requires_grad = True

# change the classifier, map the internal values to 10 output classes
model.fc = nn.Linear(512, 10)

if use_gpu:
    model.cuda()

# Find total parameters and trainable parameters(most paramaters are not trainable, which speed up training a lot)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Loss and optimizer
criteration = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
# dynamically reduce lr if loss not improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, verbose=True)

#size limit
sl = 50000
# train the model
data = get_data()
test_data = get_test_data()
X_train = data[b'data'][:sl]
X_test = test_data[b'data'][:sl]
# transpose for pytorch
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
y_train = data[b'labels'][:sl]
y_test = test_data[b'labels'][:sl]

# normalize data(this is a very important step, which increases accuracy a lot), also according to pytorch docu, image size needs to be at least 224
# https://pytorch.org/docs/stable/torchvision/models.html
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])
# create data and dataloder
data = CustomTensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), transform=transform)
test_data = CustomTensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long(), transform=transform)
dataloader = DataLoader(data, batch_size=64, shuffle=True,pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True,pin_memory=True)

reuse = False

if os.path.exists('model.pt') and reuse:
    print('Loading model')
    model = torch.load('model.pt')

else:
    print('Training model')
    # this achieves 84 accuracy just after 3 epochs, on a cpu, it takes about 3 hours per epoch
    # set model for training
    model.train()
    for epoch in range(10):
        t_loss = 0.0
        total_train_images = 0.0
        total_train_correct = 0.0
        total_test_images = 0.0
        total_test_correct = 0.0

        for data, targets in dataloader:
            if use_gpu:
                data = data.cuda()
                targets = targets.cuda()

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
            if use_gpu:
                data = data.cuda()
                targets = targets.cuda()

            # Generate predictions
            out = model(data)
            labels = torch.argmax(targets, dim=1)
            right_count = float(torch.sum(torch.argmax(out, dim=1) == labels))
            total_test_correct += right_count
            total_test_images += data.shape[0]

        model.train()
        torch.save(model, 'model.pt')
        print(
            f'{epoch} - Training Loss: {t_loss}, Training Accuracy: {total_train_correct / total_train_images}, Test Accuracy: {total_test_correct / total_test_images}')

# set model for evaluation(changes batch layers etc.)
model.eval()

# select an image to generate adversarial examples
image = np.float32(X_train[0:1])
# get the prediction of the model to that image as integer
label = int(np.argmax(model(torch.from_numpy(transform(image)).float()).detach().numpy()))

adversarial = None
# stop if unsuccessful after #timeout trials
timeout = 5
while adversarial is None and timeout >= 0:
    fmodel = PyTorchModel(model, bounds=(0, 255), num_classes=10)
    attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=TargetClass(1))
    # multiply the image with 255, to reverse the normalization before the activations
    adversarial = attack(image[0], label, verbose=True, iterations=100)
    timeout -= 1

print('Original image predicted as {}'.format(np.argmax(model(torch.from_numpy(transform(image)).float()).detach().numpy())))
print('Adversarial image predicted as {}'.format(np.argmax(model(torch.from_numpy(transform(np.array([adversarial]))).float()).detach().numpy())))

# before plotting, we need to change the the image shape and also reverse normalization.
plot_result(np.transpose(image, (0, 2, 3, 1)), np.transpose(adversarial, (1, 2, 0)))
