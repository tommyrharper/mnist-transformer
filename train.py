import torch
from torch import nn, optim
from transformer import VisionTransformer
from dataloader import four_digit_train

model = VisionTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)
    print(size)


print('training...')
train(four_digit_train, model, loss_fn, optimizer)
