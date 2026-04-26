import numpy as np
import pandas as pd

from configs.base import OUTPUT_DIR
from .plots.inference_plots import _plot_equity_curves, _plot_coverage_winrate_scatter, _plot_pvalue_bars


def generate_ticker_report(ticker: str, equities: dict, results: pd.DataFrame):
    save_dir = OUTPUT_DIR / "inference_plots" / ticker.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    _plot_equity_curves(ticker, equities, results, save_dir)
    _plot_coverage_winrate_scatter(ticker, results, save_dir)
    _plot_pvalue_bars(ticker, results, save_dir)
