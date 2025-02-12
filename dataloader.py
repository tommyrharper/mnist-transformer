import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

mnist_training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

mnist_test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 32

train_dataloader = DataLoader(mnist_training_data, batch_size=batch_size)
test_dataloader = DataLoader(mnist_test_data, batch_size=batch_size)

print('data downloaded...')

for x, y in test_dataloader:
    print(f"x/image shape: {x.shape}")
    print(f"x/image data type: {x.dtype}")
    print(f"y/label shape: {y.shape}")
    print(f"y/label data type: {y.dtype}")
    break
