import torch
from torch import nn, optim
from transformer import VisionTransformer
from dataloader import train_dataloader, test_dataloader
from tqdm import tqdm

model = VisionTransformer(num_layers=1)
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

        # for images, labels in train_dataloader:
        for images, labels in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}"):
            # TODO: remove code duplication from here and in the validation loop
            images = images.to(device)
            labels = labels.to(device)

            digit_predictions = model(images)

            loss = 0
            for i in range(4):
                loss += loss_fn(digit_predictions[i], labels[:, i])
            loss = loss / 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}"):
                images = images.to(device)
                labels = labels.to(device)

                digit_predictions = model(images)

                loss = 0
                for i in range(4):
                    loss += loss_fn(digit_predictions[i], labels[:, i])
                loss = loss / 4
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}\n')

        scheduler.step()

    torch.save(model.state_dict(), 'trained_model.pt')


print('training...')
train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, 1)
