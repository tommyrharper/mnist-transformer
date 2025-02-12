import torch
from torch import nn, optim
from transformer import VisionTransformer
from dataloader import train_dataloader, test_dataloader

model = VisionTransformer()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, num_epochs=10):
    model = model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            digit_predictions = model(images)

            # loss = 0
            # for i in range(4):
            #     loss += loss_fn(digit_predictions[i], labels[:, i])
            # loss = loss / 4

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # train_loss += loss.item()
            pass
    pass


print('training...')
train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device)
