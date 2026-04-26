import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.lstm import ModernLstmClassifier

class ModelPack:
    def __init__(self, ckpt_path, norm_path, feature_columns, dropout=0.2, num_classes=3, device=None):
        self.feature_columns = feature_columns
        self.device = device or torch.device("cpu")

        state_dict = torch.load(ckpt_path, map_location="cpu")
        arch = self._infer_arch(state_dict)

        norm_pack = torch.load(norm_path, map_location="cpu")
        self.stats_per_ticker = norm_pack["stats_per_ticker"]
        self.norm_feature_indices = norm_pack["norm_feature_indices"]

        self.model = ModernLstmClassifier(
            input_dim=arch["input_dim"],
            num_tickers=arch["num_tickers"],
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            dropout=dropout,
            num_classes=num_classes,
            ticker_emb_dim=arch["ticker_emb_dim"],
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _infer_arch(state_dict):
        emb = state_dict["ticker_emb.weight"]
        hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4

        num_layers = 0
        while f"lstm.weight_ih_l{num_layers}" in state_dict:
            num_layers += 1

        input_dim = state_dict["input_projection.0.weight"].shape[1] - emb.shape[1]

        return {
            "num_tickers": int(emb.shape[0]),
            "ticker_emb_dim": int(emb.shape[1]),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "input_dim": int(input_dim),
        }

    def _normalize_window(self, feature_window, ticker_index):
        features = torch.from_numpy(feature_window.astype(np.float32))
        stats = self.stats_per_ticker[ticker_index]
        mean, std = stats["mean"], stats["std"]

        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean.astype(np.float32))
            std = torch.from_numpy(std.astype(np.float32))

        features[:, self.norm_feature_indices] = (features[:, self.norm_feature_indices] - mean) / std
        return features

    @torch.no_grad()
    def predict_window(self, feature_window, ticker_index):
        features = self._normalize_window(feature_window, ticker_index).unsqueeze(0).to(self.device)
        ticker_tensor = torch.tensor([ticker_index], dtype=torch.long, device=self.device)
        logits = self.model(features, ticker_tensor)
        return F.softmax(logits, dim=1)[0].cpu().numpy()

    def candle_to_feature_row(self, candle):
        row = []
        for col in self.feature_columns:
            val = candle.get(col)
            if val is None:
                return None
            try:
                num = float(val)
            except Exception:
                return None
            if not math.isfinite(num):
                return None
            row.append(num)
        return row

    def predict_candles(self, candles, ticker_index, window_size, timeframe_name):
        feature_rows = [self.candle_to_feature_row(c) for c in candles]
        prediction_rows = []

        for i in range(window_size - 1, len(candles)):
            window_slice = feature_rows[i - window_size + 1:i + 1]
            if any(row is None for row in window_slice):
                continue

            probs = self.predict_window(np.asarray(window_slice, dtype=np.float32), ticker_index)
            candle = candles[i]

            prediction_rows.append({
                "i": i,
                "closeTs": candle.get("closeTs", candle.get("CloseTs", candle.get("timestamp"))),
                "close": candle.get("Close", candle.get("close")),
                "EventDirection5m": candle.get("EventDirection5m"),
                "EventDirection15m": candle.get("EventDirection15m"),
                f"pred_{timeframe_name}": int(np.argmax(probs)),
                f"p_down_{timeframe_name}": float(probs[0]),
                f"p_flat_{timeframe_name}": float(probs[1]),
                f"p_up_{timeframe_name}": float(probs[2]),
            })

        return pd.DataFrame(prediction_rows)
