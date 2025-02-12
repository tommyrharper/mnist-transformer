import wandb
import torch

class WandbLogger:
    def __init__(self, project, config):
        self.enabled = True
        wandb.init(project=project, config=config)

    def log_batch(self, loss, accuracy, is_val=False):
        prefix = "val_" if is_val else "train_"
        wandb.log({
            f"{prefix}batch_loss": loss,
            f"{prefix}batch_accuracy": accuracy
        })

    def log_epoch(self, train_loss, train_acc, val_loss, val_acc):
        wandb.log({
            "epoch_loss": {"train": train_loss, "val": val_loss},
            "epoch_accuracy": {"train": train_acc, "val": val_acc}
        })

    def save_checkpoint(self, model, epoch):
        path = f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), path)
        wandb.save(path)

    def finish(self):
        wandb.finish()