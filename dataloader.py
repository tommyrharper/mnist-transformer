import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

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

class FourDigitMNIST(Dataset):
    def __init__(self, mnist_data):
        self.minst_data = mnist_data

    def __len__(self):
        return len(self.mnist_dataset) // 4
    
    def __getitem__(self, idx):
        # Get 4 random digits from the dataset
        indices = random.sample(range(len(self.mnist_dataset)), 4)
        digits = [self.mnist_dataset[i][0] for i in indices]  # Get images
        labels = torch.tensor([self.mnist_dataset[i][1] for i in indices])  # Get labels as tensor [4,2,7,1]
        
        # Create composite image
        composite_image = torch.zeros(1, 56, 56)
        composite_image[:, :28, :28] = digits[0]
        composite_image[:, :28, 28:] = digits[1]
        composite_image[:, 28:, :28] = digits[2]
        composite_image[:, 28:, 28:] = digits[3]
        
        return composite_image, labels

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
