from .base import *

TIMEFRAME_SUBDIR = "5m"
WINDOW_SIZE = 30
TARGET_COLUMN = "EventDirection5m"
TF_SECONDS = 300

FEATURE_COLUMNS = [
    "BODYTOCLOSE",
    "HIGHTOCLOSE",
    "LOWTOCLOSE",
    "ATRTOCLOSE",
    "CloseVsSma10",
    "CloseVsSma20",
    "CloseVsSma50",
    "Sma10VsSma20",
    "Sma20VsSma50",
    "VROCLOG",
    "RSI",
    "ADXRATIO",
    "PVOH",
    "LOGRETURN",
    "CLOSETOVWAP",
    "CLOSETOAVWAP",
    "AILLIQLOG",
    "SKEW",
    "KER",
]