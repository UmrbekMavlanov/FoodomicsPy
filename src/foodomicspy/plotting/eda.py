"""
Exploratory data-analysis plots for FoodomicsPy.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.gridspec import GridSpec


def plot_y_distribution(
    y,
    y_train=None,
    y_test=None,
    *,
    bins=20,
    title=None,
    xlabel="Response",
    ylabel="Density",
    show_std=False,
    kde=True,
    figsize=(5.8, 5.0),
    full_color="lightgray",
    train_color="steelblue",
    test_color="tab:orange",
    alpha=0.55,
    edgecolor="black",
    linewidth=0.5,
    legend=True,
    strip=True,
    strip_height_ratio=0.26,
    strip_size=3,
    strip_alpha=0.75,
    random_state=42,
):
    """
    EDA plot with:
    - histogram
    - KDE
    - horizontal strip plot at bottom
    - horizontal boxplot overlaid on strip plot

    Supports:
    - full y only
    - y_train vs y_test

    Parameters
    ----------
    y : array-like
        Full dataset values.
    y_train : array-like or None, default=None
        Training-set values.
    y_test : array-like or None, default=None
        Test-set values.
    bins : int or sequence, default=20
        Number of histogram bins or explicit bin edges.
    title : str, default="Distribution of response variable"
        Figure title.
    xlabel : str, default="Response"
        X-axis label.
    ylabel : str, default="Density"
        Y-axis label.
    show_std : bool, default=False
        If True, shade mean ± std of the full y distribution.
    kde : bool, default=True
        If True, overlay KDE curve(s).
    figsize : tuple, default=(5.8, 5.0)
        Figure size in inches.
    full_color : str, default="lightgray"
        Histogram color for full dataset.
    train_color : str, default="steelblue"
        Histogram color for training set.
    test_color : str, default="tab:orange"
        Histogram color for test set.
    alpha : float, default=0.55
        Histogram transparency.
    edgecolor : str, default="black"
        Histogram bar edge color.
    linewidth : float, default=0.8
        Histogram bar edge width.
    legend : bool, default=True
        Whether to show legend in train/test case.
    strip : bool, default=True
        If True, add bottom strip + boxplot panel.
    strip_height_ratio : float, default=0.26
        Relative height of lower panel.
    strip_size : float, default=2.2
        Marker size for strip points.
    strip_alpha : float, default=0.75
        Marker transparency.
    random_state : int, default=42
        Random seed for reproducible strip jitter.

    Returns
    -------
    fig, (ax, ax_strip)

    Usage
    -----
    fig, (ax, ax_strip) = f.plotting.plot_y_distribution(
        y,
        title="Dry matter distribution",
        xlabel="Dry matter (%)",
    )
    plt.show()

    fig, (ax, ax_strip) = f.plotting.plot_y_distribution(
        y,
        y_train=y_train,
        y_test=y_test,
        title="Train vs test dry matter distribution",
        xlabel="Dry matter (%)",
    )
    plt.show()
    """
    # -------------------------
    # Style
    # -------------------------
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    rng = np.random.default_rng(random_state)

    # -------------------------
    # Prepare data
    # -------------------------
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]

    if y.size == 0:
        raise ValueError("y contains no finite values.")

    has_split = (y_train is not None) or (y_test is not None)

    if has_split:
        if y_train is None or y_test is None:
            raise ValueError("Provide both y_train and y_test, or neither.")

        y_train = np.asarray(y_train, dtype=float).ravel()
        y_test = np.asarray(y_test, dtype=float).ravel()

        y_train = y_train[np.isfinite(y_train)]
        y_test = y_test[np.isfinite(y_test)]

        if y_train.size == 0:
            raise ValueError("y_train contains no finite values.")
        if y_test.size == 0:
            raise ValueError("y_test contains no finite values.")

    # -------------------------
    # Shared bins
    # -------------------------
    if has_split:
        all_vals = np.concatenate([y_train, y_test])
    else:
        all_vals = y

    if np.isscalar(bins):
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    else:
        bin_edges = np.asarray(bins)

    # -------------------------
    # Figure and axes
    # -------------------------
    if strip:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            2, 1,
            height_ratios=[1 - strip_height_ratio, strip_height_ratio],
            hspace=0.03,
            figure=fig,
        )
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_strip = None

    # -------------------------
    # Main histogram panel
    # -------------------------
    if not has_split:
        sns.histplot(
            y,
            bins=bin_edges,
            stat="density",
            color=full_color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.85,
            ax=ax,
        )

        if kde and y.size > 1:
            sns.kdeplot(
                y,
                color="black",
                linewidth=1.8,
                ax=ax,
                clip=(bin_edges[0], bin_edges[-1]),
            )

    else:
        sns.histplot(
            y_train,
            bins=bin_edges,
            stat="density",
            color=train_color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            label="Train",
        )

        sns.histplot(
            y_test,
            bins=bin_edges,
            stat="density",
            color=test_color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            label="Test",
        )

        if kde:
            if y_train.size > 1:
                sns.kdeplot(
                    y_train,
                    color=train_color,
                    linewidth=1.8,
                    ax=ax,
                    clip=(bin_edges[0], bin_edges[-1]),
                )
            if y_test.size > 1:
                sns.kdeplot(
                    y_test,
                    color=test_color,
                    linewidth=1.8,
                    ax=ax,
                    clip=(bin_edges[0], bin_edges[-1]),
                )

        if legend:
            ax.legend(frameon=False)

    # -------------------------
    # Optional std shading
    # -------------------------
    if show_std:
        mean_val = np.mean(y)
        std_val = np.std(y, ddof=1) if y.size > 1 else 0.0
        ax.axvspan(
            mean_val - std_val,
            mean_val + std_val,
            color="0.5",
            alpha=0.08,
            zorder=0,
        )

    # -------------------------
    # Main axis styling
    # -------------------------
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    if strip:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    else:
        ax.set_xlabel(xlabel)

    # -------------------------
    # Bottom panel: boxplot + strip
    # -------------------------
    if strip and ax_strip is not None:

        if not has_split:
            # Boxplot
            ax_strip.boxplot(
                y,
                vert=False,
                positions=[1],
                widths=0.45,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1),
                medianprops=dict(color="black", linewidth=1),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                flierprops=dict(
                    marker="o",
                    markersize=4,
                    markerfacecolor="none",
                    markeredgecolor="black",
                    linestyle="none",
                ),
            )

            # Strip points
            y_jitter = rng.uniform(-0.14, 0.14, size=y.size) + 1.0
            ax_strip.scatter(
                y,
                y_jitter,
                s=strip_size**2,
                color="grey",
                alpha=strip_alpha,
                linewidth=0,
                zorder=3,
            )

            ax_strip.set_yticks([1])
            ax_strip.set_yticklabels([f"(n={len(y)})"])  
            ax_strip.set_ylim(0.6, 1.4)

        else:
            # TRAIN first, TEST second
            bp = ax_strip.boxplot(
                [y_train, y_test],
                vert=False,
                positions=[1, 2],
                widths=0.45,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1),
                medianprops=dict(color="black", linewidth=1),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                flierprops=dict(
                    marker="o",
                    markersize=4,
                    markerfacecolor="none",
                    markeredgecolor="black",
                    linestyle="none",
                ),
            )

            # Keep white fill or set colors here if wanted
            bp["boxes"][0].set_facecolor("white")  # Train
            bp["boxes"][1].set_facecolor("white")  # Test

            # Strip points aligned to same order
            y_train_jitter = rng.uniform(-0.14, 0.14, size=y_train.size) + 1.0
            y_test_jitter = rng.uniform(-0.14, 0.14, size=y_test.size) + 2.0

            ax_strip.scatter(
                y_train,
                y_train_jitter,
                s=strip_size**2,
                color=train_color,
                alpha=strip_alpha,
                linewidth=0,
                zorder=3,
            )

            ax_strip.scatter(
                y_test,
                y_test_jitter,
                s=strip_size**2,
                color=test_color,
                alpha=strip_alpha,
                linewidth=0,
                zorder=3,
            )

            # Match visual order with legend
            ax_strip.set_yticks([1, 2])
            ax_strip.set_yticklabels([
            f"(n={len(y_train)})",
            f"(n={len(y_test)})"
             ])
            ax_strip.set_ylim(0.5, 2.5)
            ax_strip.invert_yaxis()  # puts Train on top, Test below

        ax_strip.set_xlabel(xlabel)
        ax_strip.grid(False)
        ax_strip.spines["top"].set_visible(False)
        ax_strip.spines["right"].set_visible(False)
        ax_strip.spines["left"].set_visible(False)
        ax_strip.spines["bottom"].set_linewidth(0.9)

    fig.subplots_adjust(hspace=0.05)
    return fig, (ax, ax_strip)
