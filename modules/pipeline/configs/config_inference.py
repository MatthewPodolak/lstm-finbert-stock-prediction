# TICKERS = ["GOOGL", "AAPL", "META", "TSLA", "AMZN", "MSFT", "NVDA"]
TICKERS = ["META", "AAPL"]

PAYOUT = 2.0
PENALTY = 1.0
LOOKBACK_HOURS = 6.0

UP_THRESHOLD = 0.45
FLAT_KILL = 0.40
NLP_VETO_TH = 0.40
NLP_BOOST_TH = 0.60
NLP_BOOST_DELTA = 0.05
NLP_STRONG_TH = 0.70

BASELINE_STRATEGIES = ["baseline_5m", "baseline_15m", "softvote", "agreement", "gate_15m_exec_5m"]
NLP_STRATEGIES = ["nlp_veto_5m", "nlp_veto_15m", "nlp_boost_5m", "nlp_boost_15m","nlp_combined_5m", "nlp_combined_15m"]
ALL_STRATEGIES = BASELINE_STRATEGIES + NLP_STRATEGIES

STRATEGY_COLORS = {
    "baseline_5m":"#c0392b",
    "baseline_15m":"#e74c3c",
    "softvote":"#922b21",
    "agreement":"#f1948a",
    "gate_15m_exec_5m": "#1f77b4",
    "nlp_veto_5m":"#6c3483",
    "nlp_veto_15m": "#8e44ad",
    "nlp_boost_5m":"#1e8449",
    "nlp_boost_15m": "#27ae60",
    "nlp_combined_5m":"#d4ac0d",
    "nlp_combined_15m":"#f39c12",
}

def get_strategy_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#7f8c8d")