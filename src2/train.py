import torch
from torch import nn, optim
from src2.transformer import VisionTransformer
from src2.dataloader import train_dataloader, test_dataloader
from tqdm import tqdm
from src2.logger import WandbLogger
import argparse

debug_mode = False

def debug_printer(digit_predictions, position_predictions, labels, positions):
    print("\nPrediction Check:")
    for i in range(labels.size(0)):
        print(f"Sample {i}:")
        print(f"Digit - Pred: {digit_predictions[i].item()}, True: {labels[i].item()}")
        print(f"Position - Pred: {position_predictions[i].item()}, True: {positions[i].item()}")
        print(f"Correct? Digit: {digit_predictions[i] == labels[i]}, Position: {position_predictions[i] == positions[i]}")


def do_prediction_and_calculate_loss(images, labels, positions, model, device, loss_fn):
    images = images.to(device)
    labels = labels.to(device)
    positions = positions.to(device)

    digit_logits, position_logits = model(images)  # Get logits

    # Calculate losses using CrossEntropyLoss
    digit_loss = loss_fn(digit_logits, labels)  # expects [batch_size, num_classes], [batch_size]
    position_loss = loss_fn(position_logits, positions)
    loss = (digit_loss + position_loss) / 2

    # Get predictions for accuracy calculation
    digit_predictions = digit_logits.argmax(dim=1)
    position_predictions = position_logits.argmax(dim=1)

    # Calculate accuracy
    correct_digits = (digit_predictions == labels).sum().item()
    correct_positions = (position_predictions == positions).sum().item()
    total = labels.size(0)

    # Debug print for predictions
    if debug_mode:
        debug_printer(digit_predictions, position_predictions, labels, positions)

    return loss, correct_digits + correct_positions, total * 2


def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, num_epochs=10, use_wandb=False, log_epochs=False):
    model = model.to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger = WandbLogger("mnist-transformer-2", {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": len(train_dataloader),
        "epochs": num_epochs,
        "num_layers": model.num_layers,
        "weight_decay": optimizer.param_groups[0]['weight_decay']
    }) if use_wandb else None

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        correct_digits = total_digits = 0
        for images, labels, positions in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}"):

            loss, correct, total = do_prediction_and_calculate_loss(images, labels, positions, model, device, loss_fn)
            train_loss += loss.item()
            correct_digits += correct
            total_digits += total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logger: logger.log_batch(loss.item(), correct/total)

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = val_total = 0

        with torch.no_grad():
            for images, labels, positions in tqdm(test_dataloader, desc=f"Val Epoch {epoch + 1}"):
                loss, correct, total = do_prediction_and_calculate_loss(images, labels, positions, model, device, loss_fn)
                val_loss += loss.item()
                val_correct += correct
                val_total += total

                if logger: logger.log_batch(loss.item(), correct/total, is_val=True)

        # Compute epoch metrics
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(test_dataloader)
        train_acc = correct_digits / total_digits
        val_acc = val_correct / val_total

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}\n')

        if logger:
            if log_epochs: logger.log_epoch(avg_train_loss, train_acc, avg_val_loss, val_acc)
            logger.save_checkpoint(model, epoch)

        scheduler.step()

    torch.save(model.state_dict(), 'trained_model_2.pt')
    if logger: logger.finish()

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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, args.epochs, args.wandb, args.log_epochs)
