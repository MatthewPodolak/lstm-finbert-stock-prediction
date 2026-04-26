from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from configs.plots_configs import *


def plot_classification_report(y_true, y_pred, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    class_names = ["Spadek", "Flat", "Wzrost"]
    metric_names = ["precision", "recall", "f1-score", "support"]
    row_names = class_names + ["accuracy", "macro avg", "weighted avg"]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    table_data = []
    for class_name in class_names:
        table_data.append([report[class_name][m] for m in metric_names])

    accuracy_support = int(sum(report[c]["support"] for c in class_names))
    table_data.append([np.nan, np.nan, float(report["accuracy"]), accuracy_support])
    table_data.append([report["macro avg"][m] for m in metric_names])
    table_data.append([report["weighted avg"][m] for m in metric_names])

    formatted_cells = []
    for row in table_data:
        formatted_row = []
        for col_idx, value in enumerate(row):
            if col_idx == 3:
                formatted_row.append(f"{int(value)}")
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                formatted_row.append("")
            else:
                formatted_row.append(f"{float(value):.4f}")
        formatted_cells.append(formatted_row)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(cellText=formatted_cells, rowLabels=row_names, colLabels=metric_names, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_TEXT)
    table.scale(1.15, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(BLACK)
        cell.set_linewidth(LINE_WIDTH_THIN)
        if row_idx == 0 or col_idx == -1:
            cell.set_text_props(fontweight=FONT_WEIGHT_TITLE)

    plt.tight_layout()
    output_path = save_path / "classification_report.png"
    save_fig(fig, output_path)
    plt.close(fig)

    return report


def plot_reliability_diagram(y_true, y_probs, save_path, tf):
    n_bins = 10
    max_probs = np.max(y_probs, axis=1)
    pred_classes = np.argmax(y_probs, axis=1)
    is_correct = (pred_classes == y_true)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(is_correct[mask].mean())
            bin_confs.append(max_probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    valid = ~np.isnan(bin_accs)
    ece = np.sum(bin_counts[valid] * np.abs(bin_accs[valid] - bin_confs[valid])) / bin_counts[valid].sum()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DUAL)

    ax = axes[0]
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=BAR_ALPHA_LIGHT, color=BLUE)
    ax.plot([0, 1], [0, 1], color=RED, linestyle=LINE_STYLE_DASHED, label='Idealna kalibracja')
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color=RED)
    ax.set_xlabel('Pewność modelu', fontsize=FONT_LABEL)
    ax.set_ylabel('Rzeczywista dokładność', fontsize=FONT_LABEL)
    ax.set_title(f'Diagram kalibracji - MODEL {tf} - ECE = {ece:.4f}', fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.legend(loc='upper left', fontsize=FONT_TEXT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=GRID_ALPHA)

    ax = axes[1]
    ax.bar(bin_centers, bin_counts, width=0.08, alpha=BAR_ALPHA_LIGHT, color=BLUE)
    ax.set_xlabel('Pewność modelu', fontsize=FONT_LABEL)
    ax.set_ylabel('Liczba próbek', fontsize=FONT_LABEL)
    ax.set_title('Rozkład pewności predykcji', fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_fig(fig, save_path / 'ece.png')
    plt.close()

    return ece


def plot_feature_importance(model, loader, device, feature_names, save_path, seed, tf):
    import torch

    max_samples = 10000
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    all_features, all_targets, all_tickers = [], [], []
    collected = 0
    for features, targets, ticker_indices, _ in loader:
        all_features.append(features)
        all_targets.append(targets)
        all_tickers.append(ticker_indices)
        collected += features.size(0)
        if collected >= max_samples:
            break

    features_full = torch.cat(all_features, dim=0)[:max_samples].to(device)
    targets_full = torch.cat(all_targets, dim=0)[:max_samples].to(device)
    tickers_full = torch.cat(all_tickers, dim=0)[:max_samples].to(device)

    with torch.no_grad():
        baseline_acc = (model(features_full, tickers_full).argmax(dim=1) == targets_full).float().mean().item()

    importance_scores = []

    for feat_idx, feat_name in enumerate(feature_names):
        torch.manual_seed(seed + feat_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + feat_idx)

        permuted = features_full.clone()
        shuffle_idx = torch.randperm(permuted.size(0), device=device)
        permuted[:, :, feat_idx] = permuted[shuffle_idx, :, feat_idx]

        with torch.no_grad():
            perm_acc = (model(permuted, tickers_full).argmax(dim=1) == targets_full).float().mean().item()

        importance_scores.append(baseline_acc - perm_acc)

    sorted_idx = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = [importance_scores[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, max(10, len(feature_names) * 0.3)))
    ax.barh(range(len(sorted_names)), sorted_vals, color=[get_bar_color(v) for v in sorted_vals])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=FONT_ANNOT)
    ax.set_xlabel('Spadek Accuracy po permutacji', fontsize=FONT_LABEL)
    ax.set_title(f'Ważność cech - MODEL {tf} - (n={len(targets_full)})', fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.axvline(0, color=BLACK, linestyle=LINE_STYLE_SOLID, linewidth=LINE_WIDTH_THIN)
    ax.grid(True, alpha=GRID_ALPHA, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, save_path / 'feats_impp.png')
    plt.close()

    return dict(zip(feature_names, importance_scores))