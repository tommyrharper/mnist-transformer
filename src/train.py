import torch
from torch import nn, optim
from src.transformer import VisionTransformer
from src.dataloader import train_dataloader, test_dataloader
from tqdm import tqdm
import argparse
import wandb

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

def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, num_epochs=10, use_wandb=False, log_epochs=False):
    model = model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="transformer-project",
            config={
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_size": len(train_dataloader),
                "epochs": num_epochs,
                "num_layers": model.num_layers,
                "weight_decay": optimizer.param_groups[0]['weight_decay']
            }
        )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_digits = 0
        total_digits = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}")):
            loss, correct, total = calculate_loss(images, labels, model, device, loss_fn)
            train_loss += loss.item()
            correct_digits += correct
            total_digits += total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log batch metrics
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": correct / total,
                })

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = correct_digits / total_digits

        model.eval()
        val_loss = 0
        correct_digits = 0
        total_digits = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}")):
                loss, correct, total = calculate_loss(images, labels, model, device, loss_fn)
                val_loss += loss.item()
                correct_digits += correct
                total_digits += total

                # Log validation batch metrics
                if use_wandb:
                    wandb.log({
                        "val_batch_loss": loss.item(),
                        "val_batch_accuracy": correct / total
                    })

        avg_val_loss = val_loss / len(test_dataloader)
        val_accuracy = correct_digits / total_digits

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n')

        # Log metrics
        if use_wandb:
            if log_epochs:
                wandb.log({
                    "epoch_loss": {
                        "train": avg_train_loss,
                        "val": avg_val_loss
                    },
                    "epoch_accuracy": {
                        "train": train_accuracy,
                        "val": val_accuracy
                    }
                })

            # Save model checkpoint to wandb
            checkpoint_path = f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        scheduler.step()

    torch.save(model.state_dict(), 'trained_model.pt')

    # Finish wandb
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    print('training...')

    lr=1e-3
    weight_decay=1e-5

    # Check for --wandb flag and --epochs <NUM_EPOCHS> etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--log-epochs', action='store_true', help='Enable Epoch Logging')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 1)')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers to use (default: 1)')
    args = parser.parse_args()

    model = VisionTransformer(num_layers=args.layers)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if args.wandb:
        wandb.init(
            project="mnist-transformer",
            name='model-training',
            config={
                "architecture": "classic-transformer",
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "num_epochs": args.epochs,
                "layers": args.layers,
            }
        )

    train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, args.epochs, args.wandb, args.log_epochs)
