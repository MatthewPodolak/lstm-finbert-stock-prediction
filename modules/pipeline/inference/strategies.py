import numpy as np
import pandas as pd

from configs.config_inference import (
    UP_THRESHOLD, FLAT_KILL,
    NLP_VETO_TH, NLP_BOOST_TH, NLP_BOOST_DELTA, NLP_STRONG_TH
)

def get_signals(df: pd.DataFrame) -> dict:

    p_up_5m   = df["p_up_5m"].values.astype(float)
    p_flat_5m = df["p_flat_5m"].values.astype(float)

    p_up_15m   = df["p_up_15m"].values.astype(float)
    p_flat_15m = df["p_flat_15m"].values.astype(float)

    has_nlp = "sentiment_mean" in df.columns
    sentiment = (
        df["sentiment_mean"].fillna(0.5).values.astype(float)
        if has_nlp
        else np.full(len(df), 0.5)
    )

    return {
        "p_up_5m":   p_up_5m,
        "p_flat_5m": p_flat_5m,
        "p_up_15m":   p_up_15m,
        "p_flat_15m": p_flat_15m,
        "sent":      sentiment,
        "has_nlp":   has_nlp,
    }


def _is_long(p_up, p_flat, threshold=UP_THRESHOLD):
    return (p_up > threshold) & (p_flat < FLAT_KILL)


def compute_signals(signals: dict, strategy: str) -> np.ndarray:

    if strategy == "baseline_5m":
        return _is_long(signals["p_up_5m"], signals["p_flat_5m"])

    if strategy == "baseline_15m":
        return _is_long(signals["p_up_15m"], signals["p_flat_15m"])

    if strategy == "agreement":
        return (_is_long(signals["p_up_5m"],  signals["p_flat_5m"]) & _is_long(signals["p_up_15m"], signals["p_flat_15m"]))

    if strategy == "gate_15m_exec_5m":
        regime_ok = signals["p_flat_15m"] < FLAT_KILL
        return regime_ok & _is_long(signals["p_up_5m"], signals["p_flat_5m"])

    if strategy == "nlp_veto_5m":
        base = _is_long(signals["p_up_5m"], signals["p_flat_5m"])
        if not signals["has_nlp"]:
            return base
        return base & (signals["sent"] >= NLP_VETO_TH)

    if strategy == "nlp_veto_15m":
        base = _is_long(signals["p_up_15m"], signals["p_flat_15m"])
        if not signals["has_nlp"]:
            return base
        return base & (signals["sent"] >= NLP_VETO_TH)

    if strategy == "nlp_boost_5m":
        if not signals["has_nlp"]:
            return _is_long(signals["p_up_5m"], signals["p_flat_5m"])
        boosted_th = np.where(signals["sent"] > NLP_BOOST_TH, UP_THRESHOLD - NLP_BOOST_DELTA, UP_THRESHOLD)
        return (signals["p_up_5m"] > boosted_th) & (signals["p_flat_5m"] < FLAT_KILL)

    if strategy == "nlp_boost_15m":
        if not signals["has_nlp"]:
            return _is_long(signals["p_up_15m"], signals["p_flat_15m"])
        boosted_th = np.where(signals["sent"] > NLP_BOOST_TH, UP_THRESHOLD - NLP_BOOST_DELTA, UP_THRESHOLD)
        return (signals["p_up_15m"] > boosted_th) & (signals["p_flat_15m"] < FLAT_KILL)

    if strategy == "nlp_combined_5m":
        if not signals["has_nlp"]:
            return _is_long(signals["p_up_5m"], signals["p_flat_5m"])
        return ( (signals["p_up_5m"] > UP_THRESHOLD) | (signals["sent"] > NLP_STRONG_TH)) & (signals["p_flat_5m"] < FLAT_KILL)

    if strategy == "nlp_combined_15m":
        if not signals["has_nlp"]:
            return _is_long(signals["p_up_15m"], signals["p_flat_15m"])
        return (
            (signals["p_up_15m"] > UP_THRESHOLD) | (signals["sent"] > NLP_STRONG_TH)) & (signals["p_flat_15m"] < FLAT_KILL)

    if strategy == "softvote":
        p_up_avg   = 0.5 * (signals["p_up_5m"]  + signals["p_up_15m"])
        p_flat_avg = 0.5 * (signals["p_flat_5m"] + signals["p_flat_15m"])
        return _is_long(p_up_avg, p_flat_avg)

    raise ValueError(f"Unknown strategy: {strategy}")