import json
from pathlib import Path
import numpy as np
import pandas as pd
from models.finbert import FinBERTAnalyzer
from configs.config_finbert import MODEL_DIR

class NewsSentimentAnalyzer:
    def __init__(self, news_dir: Path):
        self.news_dir = Path(news_dir)
        self.finbert = FinBERTAnalyzer(MODEL_DIR)

    def _load_news(self, ticker: str) -> list:
        base_ticker = ticker.split('.')[0].lower()
        path = self.news_dir / f"news_{base_ticker}.json"
        if not path.exists():
            print(f"Brak newsow dla {ticker}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _analyze_news(self, news_items: list) -> tuple:
        timestamps = []
        scores = []

        for item in news_items:
            ts = item.get("timestamp")
            if ts is None:
                continue

            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title}. {content[:500]}" if content else title

            if not text.strip():
                continue

            try:
                result = self.finbert.analyze(text)
                timestamps.append(int(ts))
                scores.append(result["score"])
            except Exception:
                continue

        return np.array(timestamps), np.array(scores)

    def add_sentiment_to_df(self, df: pd.DataFrame, ticker: str, lookback_hours: float = 6.0, timestamp_col: str = "closeTs") -> pd.DataFrame:
        df = df.copy()
        df["sentiment_mean"] = np.nan
        df["sentiment_count"] = 0

        news_items = self._load_news(ticker)
        if not news_items:
            return df

        timestamps, scores = self._analyze_news(news_items)
        if len(timestamps) == 0:
            return df

        lookback_sec = int(lookback_hours * 3600)

        for row_idx, row in df.iterrows():
            row_ts = row.get(timestamp_col)
            if pd.isna(row_ts):
                continue

            close_ts = int(row_ts)
            mask = (timestamps >= close_ts - lookback_sec) & (timestamps <= close_ts)

            if mask.sum() == 0:
                continue

            df.at[row_idx, "sentiment_mean"] = float(scores[mask].mean())
            df.at[row_idx, "sentiment_count"] = int(mask.sum())

        return df