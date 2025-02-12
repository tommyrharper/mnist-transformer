import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

class FourDigitMNIST(datasets.MNIST):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __len__(self):
        return len(self.data) // 4
    
    def __getitem__(self, idx):
        # Get 4 random digits from the dataset
        indices = random.sample(range(len(self.data)), 4)
        digits = [self.data[i] for i in indices]  # Get images
        labels = torch.tensor([self.targets[i] for i in indices])  # Get labels as tensor [4,2,7,1]
        
        # Create composite image
        composite_image = torch.zeros(1, 56, 56)
        composite_image[:, :28, :28] = digits[0]
        composite_image[:, :28, 28:] = digits[1]
        composite_image[:, 28:, :28] = digits[2]
        composite_image[:, 28:, 28:] = digits[3]
        
        return composite_image, labels

four_digit_train = FourDigitMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
four_digit_test = FourDigitMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 32
train_dataloader = DataLoader(four_digit_train, batch_size=batch_size)
test_dataloader = DataLoader(four_digit_test, batch_size=batch_size)

# Create the four-digit dataset
print('data downloaded...')

for x, y in test_dataloader:
    print(f"x/image shape: {x.shape}")
    print(f"x/image data type: {x.dtype}")
    print(f"y/label shape: {y.shape}")
    print(f"y/label data type: {y.dtype}")
    break
