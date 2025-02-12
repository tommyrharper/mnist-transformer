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


def calculate_loss(images, labels, model, device, loss_fn):
    images = images.to(device)
    labels = labels.to(device)

    digit_predictions = model(images)

    loss = 0
    for i in range(4):
        loss += loss_fn(digit_predictions[i], labels[:, i])
    loss = loss / 4

    correct_digits = 0
    digits_checked = 0
    # Calculate accuracy
    for i in range(4):
        pred = digit_predictions[i].argmax(dim=1)
        correct_digits += (pred == labels[:, i]).sum().item()
        digits_checked += labels.size(0)

    return loss, correct_digits, digits_checked

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, num_epochs=10):
    model = model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_digits = 0
        total_digits = 0

        # for images, labels in train_dataloader:
        for images, labels in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}"):
            # TODO: remove code duplication from here and in the validation loop

            loss, correct, total = calculate_loss(images, labels, model, device, loss_fn)
            train_loss += loss.item()
            correct_digits += correct
            total_digits += total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = correct_digits / total_digits

        model.eval()
        val_loss = 0
        correct_digits = 0
        total_digits = 0

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

                # Calculate accuracy
                for i in range(4):
                    pred = digit_predictions[i].argmax(dim=1)
                    correct_digits += (pred == labels[:, i]).sum().item()
                    total_digits += labels.size(0)

        avg_val_loss = val_loss / len(test_dataloader)
        val_accuracy = correct_digits / total_digits

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n')

        scheduler.step()

    torch.save(model.state_dict(), 'trained_model.pt')


print('training...')
train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, 1)
