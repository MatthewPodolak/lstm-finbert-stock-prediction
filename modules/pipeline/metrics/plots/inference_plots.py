import matplotlib.pyplot as plt
import numpy as np
from configs.config_inference import BASELINE_STRATEGIES, STRATEGY_COLORS, get_strategy_color
from configs.plots_configs import *


def _plot_equity_curves(ticker, equities, results, save_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for strategy_name, equity_curve in equities.items():
        row = results[results["strategy"] == strategy_name]
        if len(row) > 0:
            wr = row.iloc[0]["winrate"]
            n = row.iloc[0]["trades"]
            label = f"{strategy_name} (WR={wr:.1%}, n={n})"
        else:
            label = strategy_name

        color = STRATEGY_COLORS.get(strategy_name, GRAY)
        linestyle = LINE_STYLE_DASHED if strategy_name in BASELINE_STRATEGIES else LINE_STYLE_SOLID
        ax.plot(equity_curve, label=label, color=color, linewidth=LINE_WIDTH, linestyle=linestyle, alpha=BAR_ALPHA)

    ax.axhline(y=0, color=BLACK, linestyle=LINE_STYLE_SOLID, linewidth=LINE_WIDTH_THIN, alpha=0.5)
    ax.set_ylabel("PnL", fontsize=FONT_LABEL)
    ax.set_title(f"{ticker} - Krzywe kapitału 2:1", fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.legend(loc="upper left", fontsize=FONT_TEXT - 1, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_fig(fig, save_dir / f"equity_{ticker.lower()}.png")
    plt.close()


def _plot_coverage_winrate_scatter(ticker, results, save_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)

    for _, row in results.iterrows():
        color = get_strategy_color(row["strategy"])
        ax.scatter(row["coverage"], row["winrate"],
                   s=SCATTER_SIZE, c=color, alpha=SCATTER_ALPHA,
                   edgecolors=SCATTER_EDGE_COLOR, linewidth=SCATTER_EDGE_WIDTH, zorder=5)
        ax.annotate(row["strategy"], (row["coverage"], row["winrate"]),
                    xytext=ANNOT_OFFSET, textcoords="offset points", fontsize=FONT_ANNOT)

    ax.axhline(y=0.5, color=RED, linestyle=LINE_STYLE_DASHED, linewidth=LINE_WIDTH - 0.3, alpha=0.7, label="50%")
    ax.set_xlabel("Pokrycie (Liczba transakcji)", fontsize=FONT_LABEL)
    ax.set_ylabel("Skuteczność", fontsize=FONT_LABEL)
    ax.set_title(f"{ticker} - Skuteczność", fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=FONT_TEXT - 1)
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_fig(fig, save_dir / f"coverage_winrate_scatter_{ticker.lower()}.png")
    plt.close()


def _plot_pvalue_bars(ticker, results, save_dir):
    strategies = results["strategy"].tolist()
    pv_coin = results["pvalue_vs_coin"].tolist()
    colors = [get_strategy_color(s) for s in strategies]
    x = np.arange(len(strategies))

    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.1), 6))
    bars = ax.bar(x, pv_coin, color=colors, alpha=SCATTER_ALPHA, edgecolor=BLACK, linewidth=BAR_EDGE_WIDTH)

    for bar, val in zip(bars, pv_coin):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)

    ax.axhline(y=0.05, color=RED, linestyle=LINE_STYLE_DASHED, linewidth=LINE_WIDTH, alpha=0.8, label="α = 0.05")
    ax.set_ylabel("p", fontsize=FONT_LABEL)
    ax.set_title(f"{ticker} - Istotność statystyczna", fontsize=FONT_TITLE, fontweight=FONT_WEIGHT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=XTICK_ROTATION, ha=XTICK_HA, fontsize=FONT_TEXT - 1)
    ax.set_ylim(0, min(1.05, max(pv_coin) * 1.15 + 0.1))
    ax.legend(fontsize=FONT_TEXT)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    plt.tight_layout()
    save_fig(fig, save_dir / f"pvalue_bars_{ticker.lower()}.png")
    plt.close()