import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train=True,
    download=False,
    transform=transforms.Compose(
    [transforms.ToTensor()]))

image, label = train_set[0]

print (image.shape)