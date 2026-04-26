import math
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_candles

class MultiTickerDataset(Dataset):
    def __init__(self, root_dir, split, window_size, feature_columns, target_column, tf_seconds, timeframe_subdir, ticker_vocab, num_classes=3, stats_per_ticker=None, norm_feature_indices=None):
        self.window = window_size
        self.num_classes = num_classes
        self.class_counts = np.zeros(num_classes, dtype=np.int64)

        self.ticker2idx = dict(ticker_vocab)
        self.idx2ticker = {v: k for k, v in self.ticker2idx.items()}

        self.ticker_data = {}
        self.samples = []

        self._load_all(root_dir, split, feature_columns, target_column, tf_seconds, timeframe_subdir, stats_per_ticker, norm_feature_indices)

    def _parse_candles(self, candles, feature_columns, target_column, tf_seconds):
        features = []
        targets = []
        timestamps = []

        for candle in candles:
            raw_target = candle.get(target_column)
            targets.append(float("nan") if raw_target is None else int(raw_target) + 1)

            row = []
            row_ok = True
            for col in feature_columns:
                val = candle.get(col)
                if val is None:
                    row_ok = False
                    break
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    row_ok = False
                    break
                if math.isnan(num):
                    row_ok = False
                    break
                row.append(num)

            features.append(row if row_ok else [float("nan")] * len(feature_columns))

            ts = candle.get("timestamp")
            timestamps.append(int(ts) + tf_seconds if ts is not None else -1)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(timestamps, dtype=torch.int64),
        )

    def _normalize(self, features, stats, columns_to_norm):
        mean = stats["mean"]
        std = stats["std"]
        
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean.astype(np.float32))
            std = torch.from_numpy(std.astype(np.float32))

        features[:, columns_to_norm] = (features[:, columns_to_norm] - mean) / std
        return features

    def _load_all(self, root_dir, split, feature_columns, target_column, tf_seconds,
                  timeframe_subdir, stats_per_ticker, columns_to_norm):
        ticker_count = 0

        for ticker_dir in sorted(root_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue

            split_file = ticker_dir / timeframe_subdir / f"{split}.txt"
            if not split_file.exists():
                continue

            ticker_name = ticker_dir.name
            if ticker_name not in self.ticker2idx:
                continue

            ticker_idx = self.ticker2idx[ticker_name]
            candles = load_candles(split_file)

            feats, targs, times = self._parse_candles(candles, feature_columns, target_column, tf_seconds)

            if stats_per_ticker and columns_to_norm and ticker_idx in stats_per_ticker:
                feats = self._normalize(feats, stats_per_ticker[ticker_idx], columns_to_norm)

            self.ticker_data[ticker_idx] = (feats, targs, times)
            ticker_count += 1

            num_candles = len(targs)
            if num_candles < self.window:
                continue

            for end_pos in range(self.window - 1, num_candles):
                target_val = targs[end_pos].item()
                if math.isnan(target_val):
                    continue

                window = feats[end_pos - self.window + 1:end_pos + 1]
                if torch.isnan(window).any():
                    continue

                target_class = int(target_val)
                if 0 <= target_class < self.num_classes:
                    self.class_counts[target_class] += 1

                self.samples.append((ticker_idx, end_pos))

    def get_class_counts(self):
        return self.class_counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker_idx, end_pos = self.samples[idx]
        feats, targs, times = self.ticker_data[ticker_idx]

        window = feats[end_pos - self.window + 1:end_pos + 1]
        target = int(targs[end_pos].item())
        close_ts = int(times[end_pos].item())

        return window, torch.tensor(target, dtype=torch.long), ticker_idx, max(close_ts, end_pos)