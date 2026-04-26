import numpy as np
import pandas as pd
from scipy.stats import binomtest

from utils import get_ticker_idx, load_json_or_jsonl, align_5m_15m
from metrics.inference_report import generate_ticker_report
from configs import config_5m, config_15m
from configs.base import DEVICE, DROPOUT, NUM_CLASSES, OUTPUT_DIR, NEWS_DIR
from inference.model_pack import ModelPack
from configs.config_inference import ALL_STRATEGIES, NLP_STRATEGIES, TICKERS, PAYOUT, PENALTY, LOOKBACK_HOURS
from inference.strategies import get_signals, compute_signals
from data.sentiment_integration import NewsSentimentAnalyzer

def get_equity_with_metrics(actual_labels, long_signal, strategy):
    num_bars = len(actual_labels)
    per_bar_scores = np.zeros(num_bars)
    num_trades = num_wins = 0

    for i in range(num_bars):
        if not long_signal[i]:
            continue
        actual = actual_labels.iloc[i]
        if pd.isna(actual) or actual not in (-1, 1):
            continue

        num_trades += 1
        if int(actual) == 1:
            per_bar_scores[i] = PAYOUT
            num_wins += 1
        else:
            per_bar_scores[i] = -PENALTY

    equity_curve = np.cumsum(per_bar_scores)
    total_pnl = float(equity_curve[-1]) if num_bars > 0 else 0.0
    winrate = num_wins / num_trades if num_trades > 0 else 0.0
    pv_coin = binomtest(num_wins, num_trades, p=0.5, alternative="greater").pvalue if num_trades >= 2 else 1.0

    metrics = {
        "strategy": strategy,
        "trades": num_trades,
        "wins": num_wins,
        "winrate": winrate,
        "pnl": total_pnl,
        "pvalue_vs_coin": pv_coin,
        "coverage": num_trades,
    }

    return equity_curve, metrics


def test_ticker(ticker, model_pack_5m, model_pack_15m):
    nlp_strategy_set = set(NLP_STRATEGIES)

    path_5m = OUTPUT_DIR / f"final_test_{ticker.lower()}_5m.txt"
    path_15m = OUTPUT_DIR / f"final_test_{ticker.lower()}_15m.txt"
    if not path_5m.exists() or not path_15m.exists():
        return pd.DataFrame(), {}

    candles_5m = load_json_or_jsonl(path_5m)
    candles_15m = load_json_or_jsonl(path_15m)

    ticker_index = get_ticker_idx(ticker)

    df_5m = model_pack_5m.predict_candles(candles_5m, ticker_index, config_5m.WINDOW_SIZE, "5m")
    df_15m = model_pack_15m.predict_candles(candles_15m, ticker_index, config_15m.WINDOW_SIZE, "15m")
    merged = align_5m_15m(df_5m, df_15m, ticker=f"{ticker}.US")

    if len(merged) == 0:
        return pd.DataFrame(), {}

    has_nlp = False
    try:
        analyzer = NewsSentimentAnalyzer(NEWS_DIR)
        merged = analyzer.add_sentiment_to_df(merged, ticker, LOOKBACK_HOURS)
        has_nlp = "sentiment_mean" in merged.columns and merged["sentiment_count"].sum() > 0
    except Exception as e:
        print(f"--- NLP not available --- {e}")

    signals = get_signals(merged)
    label_col = "EventDirection5m_5m" if "EventDirection5m_5m" in merged.columns else "EventDirection5m"
    actual_labels = pd.to_numeric(merged[label_col], errors="coerce")

    strategy_results = []
    equity_curves = {}

    for strategy_name in ALL_STRATEGIES:
        if strategy_name in nlp_strategy_set and not has_nlp:
            continue

        long_signal = compute_signals(signals, strategy_name)
        equity_curve, metrics = get_equity_with_metrics(actual_labels, long_signal, strategy_name)

        equity_curves[strategy_name] = equity_curve
        strategy_results.append({"ticker": ticker, **metrics})

    return pd.DataFrame(strategy_results), equity_curves


def inference():
    model_pack_5m = ModelPack(
        OUTPUT_DIR / "best_attn_lstm_5m.pt",
        OUTPUT_DIR / "norm_stats_5m.pt",
        config_5m.FEATURE_COLUMNS,
        DROPOUT, NUM_CLASSES, DEVICE,
    )
    model_pack_15m = ModelPack(
        OUTPUT_DIR / "best_attn_lstm_15m.pt",
        OUTPUT_DIR / "norm_stats_15m.pt",
        config_15m.FEATURE_COLUMNS,
        DROPOUT, NUM_CLASSES, DEVICE,
    )

    for ticker in TICKERS:
        results, equity_curves = test_ticker(ticker, model_pack_5m, model_pack_15m)
        if len(results) > 0:
            generate_ticker_report(ticker, equity_curves, results)
