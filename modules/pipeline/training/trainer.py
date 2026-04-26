import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from configs.base import *
from training.early_stopping import EarlyStopping
from training.evaluation import eval_epoch
from data import MultiTickerDataset, build_ticker_vocab
from models.lstm import ModernLstmClassifier
from utils import compute_feature_stats_per_ticker, line_gen
from metrics import generate_thesis_report


def train_one_epoch(model, loader, optimizer, loss_fn, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets, ticker_indices, _ in loader:
        features = features.to(device)
        targets = targets.to(device)
        ticker_indices = ticker_indices.to(device).long()

        optimizer.zero_grad()
        loss = loss_fn(model(features, ticker_indices), targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples


def train_model(cfg):
    timeframe = cfg.TIMEFRAME_SUBDIR
    line_gen(f"TRAINING MODEL FOR - {timeframe}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    vocab_path = OUTPUT_DIR / "ticker_vocab.json"
    ticker_vocab = build_ticker_vocab(ROOT_DIR, vocab_path, timeframe)
    num_tickers = max(ticker_vocab.values()) + 1

    line_gen("BUILDING DATASETS")

    train_raw = MultiTickerDataset(
        root_dir=ROOT_DIR,
        split="training",
        window_size=cfg.WINDOW_SIZE,
        feature_columns=cfg.FEATURE_COLUMNS,
        target_column=cfg.TARGET_COLUMN,
        tf_seconds=cfg.TF_SECONDS,
        timeframe_subdir=timeframe,
        num_classes=NUM_CLASSES,
        ticker_vocab=ticker_vocab,
    )

    stats_per_ticker, norm_feature_indices = compute_feature_stats_per_ticker(train_raw)

    torch.save({"stats_per_ticker": stats_per_ticker, "norm_feature_indices": norm_feature_indices}, OUTPUT_DIR / f"norm_stats_{timeframe}.pt")

    def make_dataset(split):
        return MultiTickerDataset(
            root_dir=ROOT_DIR,
            split=split,
            window_size=cfg.WINDOW_SIZE,
            feature_columns=cfg.FEATURE_COLUMNS,
            target_column=cfg.TARGET_COLUMN,
            tf_seconds=cfg.TF_SECONDS,
            timeframe_subdir=timeframe,
            num_classes=NUM_CLASSES,
            stats_per_ticker=stats_per_ticker,
            norm_feature_indices=norm_feature_indices,
            ticker_vocab=ticker_vocab,
        )

    train_dataset = make_dataset("training")
    val_dataset = make_dataset("validation")
    test_dataset = make_dataset("test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    class_counts = train_dataset.get_class_counts()
    total_samples = class_counts.sum()
    weights_np = total_samples / (NUM_CLASSES * (class_counts + 1e-6))
    class_weights = torch.tensor(weights_np, dtype=torch.float32).to(DEVICE)
    print(f"--- Class Weights: {class_weights.cpu().numpy()} ---")

    model = ModernLstmClassifier(
        input_dim=len(cfg.FEATURE_COLUMNS),
        num_tickers=num_tickers,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        ticker_emb_dim=TICKER_EMB_DIM,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    checkpoint_path = OUTPUT_DIR / f"best_attn_lstm_{timeframe}.pt"
    early_stopper = EarlyStopping(save_path=checkpoint_path, patience=PATIENCE)

    line_gen("TRAINING STARTED")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_acc, val_pred_dist, val_targ_dist = eval_epoch(model, val_loader, loss_fn, DEVICE)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")
        print(f"  Val pred dist: {val_pred_dist}")
        print(f"  Val targ dist: {val_targ_dist}")

        scheduler.step(val_loss)
        early_stopper(val_loss, model)

        if early_stopper.early_stop:
            line_gen("BREAK - EARLY STOPPED")
            break


    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        line_gen("LOADED BEST MODEL")

        test_loss, test_acc, _, _ = eval_epoch(model, test_loader, loss_fn, DEVICE)
        print(f"TEST SET | Loss: {test_loss:.6f} | Acc: {test_acc:.2f}%")

        generate_thesis_report(
            model=model,
            loader=test_loader,
            device=DEVICE,
            feature_names=cfg.FEATURE_COLUMNS,
            timeframe_subdir=timeframe
        )
    else:
        print("NO CHECKPOINT. SKIP.")

    line_gen(f"COMPLETED FOR - {timeframe}")
