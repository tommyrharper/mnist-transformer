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
        indice = random.sample(range(len(self.data)), 1)
        digit = self.data[indice]  # Get image
        label = torch.tensor(self.targets[indice])  # Get label as tensor [3]
        

        position = random.randint(0, 3)
        # Create composite image
        composite_image = torch.zeros(1, 56, 56)

        if position == 0: composite_image[:, :28, :28] = digit
        elif position == 1: composite_image[:, :28, 28:] = digit
        elif position == 2: composite_image[:, 28:, :28] = digit
        elif position == 3: composite_image[:, 28:, 28:] = digit
        
        return composite_image, label, position

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

if __name__ == "__main__":
    # Create the four-digit dataset
    print('data downloaded...')

    for x, y, z in test_dataloader:
        print(f"x/image shape: {x.shape}")
        print(f"x/image data type: {x.dtype}")
        print(f"y/label shape: {y.shape}")
        print(f"y/label data type: {y.dtype}")
        print(f"z/position: {z}")

        print("\nPrinting individual elements in batch:")
        for i in range(batch_size):
            print(f"Sample {i}:")
            print(f"Image shape: {x[i].shape}")
            print(f"Label: {y[i].item()}")
            print(f"Position: {z[i].item()}\n")
            
        break
