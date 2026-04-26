import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

def eval_epoch(model: nn.Module, loader, loss_fn, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, ticker_indices, _ in loader:
            features = features.to(device)
            targets = targets.to(device)
            ticker_indices = ticker_indices.to(device).long()

            logits = model(features, ticker_indices)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            all_preds.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    pred_labels = preds.argmax(axis=1)

    accuracy = accuracy_score(targets, pred_labels) * 100.0

    unique_p, counts_p = np.unique(pred_labels, return_counts=True)
    unique_t, counts_t = np.unique(targets, return_counts=True)
    pred_dist = {int(k): int(v) for k, v in zip(unique_p, counts_p)}
    targ_dist = {int(k): int(v) for k, v in zip(unique_t, counts_t)}

    return total_loss / total_samples, accuracy, pred_dist, targ_dist
