from .base import *

TIMEFRAME_SUBDIR = "15m"
WINDOW_SIZE = 20
TARGET_COLUMN = "EventDirection15m"
TF_SECONDS = 900

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
    "MASSI",
    "LOGRETURN",
    "CLOSETOVWAP",
    "CLOSETOAVWAP",
    "AILLIQLOG",
    "SKEW",
    "KER",
]