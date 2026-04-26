import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, save_path: str, patience: int = 3):
        self.save_path = save_path
        self.patience = patience
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.epochs_without_improvement += 1
            print(f"EarlyStopping --> {self.epochs_without_improvement}/{self.patience}")
            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True
