from pathlib import Path
from configs.base import OUTPUT_DIR, SEED
import numpy as np
import torch
import torch.nn.functional as F

from .plots.training_plots import plot_reliability_diagram, plot_feature_importance, plot_classification_report


def generate_thesis_report(model, loader, device, feature_names, timeframe_subdir):
    save_dir = Path(OUTPUT_DIR / f"plots_{timeframe_subdir}")
    save_dir.mkdir(exist_ok=True)

    model.eval()
    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for features, targets, ticker_indices, _ in loader:
            features = features.to(device)
            ticker_indices = ticker_indices.to(device).long()
            probs = F.softmax(model(features, ticker_indices), dim=1)

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_true.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    plot_classification_report(y_true, y_pred, save_dir)
    plot_reliability_diagram(y_true, y_probs, save_dir, timeframe_subdir)
    plot_feature_importance(model, loader, device, feature_names, save_dir, SEED, timeframe_subdir)
