import torch
from torch import nn, optim
from transformer import VisionTransformer
from dataloader import train_dataloader, test_dataloader

model = VisionTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_epochs=10):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()

    for epoch in range(num_epochs):
        pass
    pass


print('training...')
train(train_dataloader, test_dataloader, model, loss_fn, optimizer)
