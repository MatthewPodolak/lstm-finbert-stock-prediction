import numpy as np
import pandas as pd
from pathlib import Path
import torch
import json

from configs.base import VOCAB_PATH


def load_candles(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)

def compute_feature_stats_per_ticker(train_dataset) -> tuple:
    stats_per_ticker = {}

    for ticker_idx, (features_tensor, _, _) in train_dataset.ticker_data.items():
        if features_tensor.numel() == 0:
            continue

        valid_mask = ~torch.isnan(features_tensor)
        valid_count = torch.clamp(valid_mask.sum(dim=0), min=1)
        features_safe = torch.where(valid_mask, features_tensor, torch.zeros_like(features_tensor))
        mean = features_safe.sum(dim=0) / valid_count
        variance = (features_safe ** 2).sum(dim=0) / valid_count - mean ** 2
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        stats_per_ticker[ticker_idx] = {"mean": mean, "std": std}

    norm_indices = list(range(features_tensor.shape[1]))
    return stats_per_ticker, norm_indices


_TICKER_VOCAB = None

def _load_ticker_vocab():
    global _TICKER_VOCAB
    if _TICKER_VOCAB is None:
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"Ticker vocab not found at {VOCAB_PATH}. Run training first.")
        with open(VOCAB_PATH, "r") as f:
            _TICKER_VOCAB = json.load(f)
    return _TICKER_VOCAB


def get_ticker_idx(ticker: str) -> int:
    vocab = _load_ticker_vocab()
    return vocab.get(f"{ticker}.US", 0)


def load_json_or_jsonl(path: Path) -> list:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        pass
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def align_5m_15m(df_5m: pd.DataFrame, df_15m: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    df_5m = df_5m.copy()
    df_15m = df_15m.copy()

    if "ticker" not in df_5m.columns and ticker is not None:
        df_5m["ticker"] = ticker
    if "ticker" not in df_15m.columns and ticker is not None:
        df_15m["ticker"] = ticker

    df_5m["closeTs"] = pd.to_numeric(df_5m["closeTs"], errors="coerce")
    df_15m["closeTs"] = pd.to_numeric(df_15m["closeTs"], errors="coerce")

    df_5m = df_5m.dropna(subset=["closeTs"] + (["ticker"] if "ticker" in df_5m.columns else []))
    df_15m = df_15m.dropna(subset=["closeTs"] + (["ticker"] if "ticker" in df_15m.columns else []))

    df_5m["closeTs"] = df_5m["closeTs"].astype(np.int64)
    df_15m["closeTs"] = df_15m["closeTs"].astype(np.int64)

    df_5m = df_5m[df_5m["closeTs"] % 900 == 0].copy()

    merge_keys = ["ticker", "closeTs"] if ("ticker" in df_5m.columns and "ticker" in df_15m.columns) else ["closeTs"]
    return df_5m.merge(df_15m, on=merge_keys, how="inner", suffixes=("_5m", "_15m"))


def line_gen(text: str):
    print(f"\n{'=' * 70}")
    print(text)
    print(f"{'=' * 70}")