"""
Statistical plots for FoodomicsPy.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def plot_correlation_bubbles(
    df,
    figsize=(10, 8),
    zero_threshold=0.0,
    min_size=20,
    max_size=480,
    palette=None,
    legend_values=(-1, -0.5, 0, 0.5, 1),
    title=None,
    correction_method="fdr_bh",
    alpha=0.05,
    annotate="strong",
    annotation_r_threshold=0.90,
    show_significance_stars=False,
    bubble_axis_padding=0.50,
    y_tick_padding=5,
    annotation_fontsize=8,
    star_fontsize=8,
    ax=None,
):
    """
    Plot the lower triangle of a Pearson correlation matrix as bubbles.

    Bubble color represents the Pearson correlation coefficient.
    Bubble size represents the absolute correlation strength.

    For each variable pair, the function calculates:

    - Pearson correlation coefficient, r
    - raw p-value
    - multiple-testing-adjusted p-value
    - significance stars based on the adjusted p-value

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with observations in rows and variables in columns.

    figsize : tuple of float, default=(10, 8)
        Figure width and height in inches.

        This is used only when ax is None.

    zero_threshold : float, default=0.0
        Hide bubbles whose absolute correlation is less than or equal
        to this value.

        This controls which bubbles are displayed. It does not control
        which bubbles are annotated.

    min_size : float, default=20
        Minimum bubble area in points squared.

    max_size : float, default=480
        Maximum bubble area in points squared.

    palette : matplotlib colormap, str, or None, default=None
        Colormap used for correlation coefficients.

        When None, a custom blue-white-wine diverging colormap is used.

    legend_values : sequence of float, default=(-1, -0.5, 0, 0.5, 1)
        Correlation values shown in the combined color-and-size legend.

    title : str or None, default=None
        Optional title displayed above the plot.

    correction_method : str, default="fdr_bh"
        Multiple-testing correction passed to
        statsmodels.stats.multitest.multipletests.

        Common options:

        - "fdr_bh": Benjamini-Hochberg FDR
        - "bonferroni": Bonferroni correction
        - "holm": Holm correction

    alpha : float, default=0.05
        Significance threshold for adjusted p-values.

    annotate : {"none", "strong"}, default="strong"
        Controls numerical annotations.

        - "none": no correlation values are displayed
        - "strong": display r for correlations satisfying
          abs(r) >= annotation_r_threshold

    annotation_r_threshold : float, default=0.90
        Minimum absolute correlation required for annotation.

        For example, 0.90 annotates correlations where:

        - r >= 0.90
        - r <= -0.90

    show_significance_stars : bool, default=False
        Add significance stars above annotated correlation values.

        Stars are based on adjusted p-values:

        - *   q < 0.05
        - **  q < 0.01
        - *** q < 0.001

    bubble_axis_padding : float, default=0.70
        Extra horizontal space added to the left side of the axes.

        This prevents the first bubble column from overlapping the
        y-axis metabolite labels, particularly in SVG files and
        narrow subplot panels.

        Increase this value for larger bubbles or narrow panels.

    y_tick_padding : float, default=6
        Distance in points between y-axis labels and the plot area.

    annotation_fontsize : float, default=8
        Font size of annotated correlation coefficients.

    star_fontsize : float, default=8
        Font size of significance stars.

    ax : matplotlib.axes.Axes or None, default=None
        Existing axes on which to draw the plot.

        When None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.

    ax : matplotlib.axes.Axes
        Matplotlib axes object.

    corr_long : pandas.DataFrame
        Long-format statistical results containing:

        - variable_y
        - variable_x
        - correlation
        - abs_correlation
        - p_value
        - p_fdr
        - significant
        - star
        - n

    Examples
    --------
    Create one standalone figure:

    >>> fig, ax, corr_data = f.plotting.plot_correlation_bubbles(
    ...     df,
    ...     figsize=(10, 8),
    ...     annotation_r_threshold=0.90,
    ...     show_significance_stars=True,
    ...     title="Mimicking with cottonseed oil",
    ... )
    >>> plt.show()

    Increase spacing between labels and the first bubble column:

    >>> fig, ax, corr_data = f.plotting.plot_correlation_bubbles(
    ...     df,
    ...     bubble_axis_padding=0.90,
    ...     y_tick_padding=8,
    ... )
    >>> plt.show()

    Use the function inside a subplot:

    >>> fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    >>>
    >>> f.plotting.plot_correlation_bubbles(
    ...     df,
    ...     ax=axs[0, 0],
    ...     annotation_r_threshold=0.90,
    ...     show_significance_stars=True,
    ... )
    >>>
    >>> fig.tight_layout()
    >>> plt.show()

    Plot without annotations:

    >>> fig, ax, corr_data = f.plotting.plot_correlation_bubbles(
    ...     df,
    ...     annotate="none",
    ... )
    >>> plt.show()

    Inspect significant correlations:

    >>> significant_results = (
    ...     corr_data
    ...     .query("significant == True")
    ...     .sort_values("abs_correlation", ascending=False)
    ... )

    Save results:

    >>> corr_data.to_excel(
    ...     "correlation_results.xlsx",
    ...     index=False,
    ... )
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not 0 <= zero_threshold <= 1:
        raise ValueError("zero_threshold must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if not 0 <= annotation_r_threshold <= 1:
        raise ValueError(
            "annotation_r_threshold must be between 0 and 1."
        )

    if min_size <= 0:
        raise ValueError("min_size must be greater than 0.")

    if max_size <= min_size:
        raise ValueError("max_size must be greater than min_size.")

    if bubble_axis_padding < 0:
        raise ValueError(
            "bubble_axis_padding must be greater than or equal to 0."
        )

    if y_tick_padding < 0:
        raise ValueError(
            "y_tick_padding must be greater than or equal to 0."
        )

    if annotation_fontsize <= 0:
        raise ValueError(
            "annotation_fontsize must be greater than 0."
        )

    if star_fontsize <= 0:
        raise ValueError(
            "star_fontsize must be greater than 0."
        )

    valid_annotations = {"none", "strong"}

    if annotate not in valid_annotations:
        raise ValueError(
            f"annotate must be one of {valid_annotations}."
        )

    if not isinstance(show_significance_stars, bool):
        raise TypeError(
            "show_significance_stars must be True or False."
        )

    # -------------------------------------------------------------------------
    # Select numerical columns
    # -------------------------------------------------------------------------
    df_numeric = df.select_dtypes(include="number").copy()

    if df_numeric.shape[1] < 2:
        raise ValueError(
            "The DataFrame must contain at least two numerical columns."
        )

    # Remove constant columns because Pearson correlation is undefined
    constant_columns = df_numeric.columns[
        df_numeric.nunique(dropna=True) <= 1
    ].tolist()

    if constant_columns:
        df_numeric = df_numeric.drop(columns=constant_columns)

    if df_numeric.shape[1] < 2:
        raise ValueError(
            "At least two non-constant numerical columns are required."
        )

    variables = df_numeric.columns.tolist()

    # -------------------------------------------------------------------------
    # Calculate Pearson r and raw p-values
    # -------------------------------------------------------------------------
    results = []

    for i in range(1, len(variables)):

        variable_y = variables[i]

        for j in range(i):

            variable_x = variables[j]

            pair_data = df_numeric[
                [variable_x, variable_y]
            ].dropna()

            n = len(pair_data)

            if n < 3:
                r_value = np.nan
                p_value = np.nan

            else:
                try:
                    r_value, p_value = pearsonr(
                        pair_data[variable_x],
                        pair_data[variable_y],
                    )

                except ValueError:
                    r_value = np.nan
                    p_value = np.nan

            results.append(
                {
                    "variable_y": variable_y,
                    "variable_x": variable_x,
                    "correlation": r_value,
                    "p_value": p_value,
                    "n": n,
                }
            )

    corr_long = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Multiple-testing correction
    # -------------------------------------------------------------------------
    corr_long["p_fdr"] = np.nan
    corr_long["significant"] = False

    valid_p = corr_long["p_value"].notna()

    if valid_p.any():

        reject, p_adjusted, _, _ = multipletests(
            corr_long.loc[valid_p, "p_value"],
            alpha=alpha,
            method=correction_method,
        )

        corr_long.loc[valid_p, "p_fdr"] = p_adjusted
        corr_long.loc[valid_p, "significant"] = reject

    # -------------------------------------------------------------------------
    # Significance stars based on adjusted p-values
    # -------------------------------------------------------------------------
    def p_to_star(p_value):

        if pd.isna(p_value):
            return ""

        if p_value < 0.001:
            return "***"

        if p_value < 0.01:
            return "**"

        if p_value < 0.05:
            return "*"

        return ""

    corr_long["star"] = corr_long["p_fdr"].apply(p_to_star)
    corr_long["abs_correlation"] = corr_long["correlation"].abs()

    # -------------------------------------------------------------------------
    # Apply bubble-display threshold
    # -------------------------------------------------------------------------
    corr_plot = corr_long.loc[
        corr_long["abs_correlation"] > zero_threshold
    ].copy()

    if corr_plot.empty:
        raise ValueError(
            "No correlations remain after applying zero_threshold."
        )

    # -------------------------------------------------------------------------
    # Preserve original variable order
    # -------------------------------------------------------------------------
    for column in ["variable_x", "variable_y"]:

        corr_plot[column] = pd.Categorical(
            corr_plot[column],
            categories=variables,
            ordered=True,
        )

    # -------------------------------------------------------------------------
    # Format numerical annotations
    # -------------------------------------------------------------------------
    def format_correlation(value):
        """
        Convert:
        0.98  -> .98
        -0.92 -> -.92
        1.00  -> 1.00
        -1.00 -> -1.00
        """

        formatted = f"{value:.2f}"

        if formatted.startswith("-0"):
            return "-" + formatted[2:]

        if formatted.startswith("0"):
            return formatted[1:]

        return formatted

    corr_plot["annotation"] = ""

    if annotate == "strong":

        strong_mask = (
            corr_plot["abs_correlation"]
            >= annotation_r_threshold
        )

        corr_plot.loc[
            strong_mask,
            "annotation",
        ] = corr_plot.loc[
            strong_mask,
            "correlation",
        ].apply(format_correlation)

    # -------------------------------------------------------------------------
    # Create colormap
    # -------------------------------------------------------------------------
    if palette is None:

        palette = LinearSegmentedColormap.from_list(
            "foodomics_correlation",
            [
                "#3F6F9F",
                "#94AEC8",
                "#DDE4EA",
                "#F7F7F7",
                "#E9C5C0",
                "#CB746B",
                "#9C2113",
            ],
        )

    elif isinstance(palette, str):

        palette = plt.get_cmap(palette)

    norm = Normalize(vmin=-1, vmax=1)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    created_new_axes = ax is None

    with sns.axes_style("whitegrid"):

        if created_new_axes:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        sns.scatterplot(
            data=corr_plot,
            x="variable_x",
            y="variable_y",
            hue="correlation",
            size="abs_correlation",
            palette=palette,
            hue_norm=(-1, 1),
            sizes=(min_size, max_size),
            size_norm=(0, 1),
            marker="o",
            edgecolor="#C9C9C9",
            linewidth=0.75,
            clip_on=True,
            legend=False,
            ax=ax,
        )

        # ---------------------------------------------------------------------
        # Add extra space between labels and the first bubble column
        # ---------------------------------------------------------------------
        n_variables = len(variables)

        ax.set_xlim(
            -0.5 - bubble_axis_padding,
            n_variables - 0.5,
        )

        # ---------------------------------------------------------------------
        # Add white correlation values and optional stars
        # ---------------------------------------------------------------------
        for _, row in corr_plot.iterrows():

            if row["annotation"] == "":
                continue

            # Correlation coefficient slightly below the bubble center
            ax.annotate(
                row["annotation"],
                xy=(row["variable_x"], row["variable_y"]),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                fontweight="bold",
                color="white",
                clip_on=True,
            )

            # Significance stars slightly above the coefficient
            if show_significance_stars and row["star"]:

                ax.annotate(
                    row["star"],
                    xy=(row["variable_x"], row["variable_y"]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=star_fontsize,
                    fontweight="bold",
                    color="white",
                    clip_on=True,
                )

        # ---------------------------------------------------------------------
        # Axis formatting
        # ---------------------------------------------------------------------
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(
            axis="x",
            labelrotation=90,
            labelsize=10,
            length=0,
        )

        ax.tick_params(
            axis="y",
            labelsize=10,
            length=0,
            pad=y_tick_padding,
        )

        ax.grid(
            visible=True,
            color="#E5E5E5",
            linewidth=0.65,
        )

        ax.set_axisbelow(True)

        sns.despine(
            ax=ax,
            left=True,
            bottom=True,
        )

        if title is not None:

            ax.set_title(
                title,
                fontsize=13,
                pad=14,
            )

        # ---------------------------------------------------------------------
        # Combined color-and-size legend
        # ---------------------------------------------------------------------
        legend_handles = []

        for value in legend_values:

            if not -1 <= value <= 1:
                raise ValueError(
                    "All legend_values must be between -1 and 1."
                )

            bubble_area = (
                min_size
                + abs(value) * (max_size - min_size)
            )

            marker_size = np.sqrt(bubble_area)

            legend_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=marker_size,
                    markerfacecolor=palette(norm(value)),
                    markeredgecolor="#C9C9C9",
                    markeredgewidth=0.75,
                    label=f"{value:.1f}",
                )
            )

        correlation_legend = ax.legend(
            handles=legend_handles,
            title="",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            borderaxespad=0,
            labelspacing=1.5,
            handletextpad=1.0,
        )

        ax.add_artist(correlation_legend)

        # ---------------------------------------------------------------------
        # Optional significance explanation
        # ---------------------------------------------------------------------
        if show_significance_stars:

            ax.text(
                1.02,
                0.55,
                "FDR significance\n"
                "*   q < 0.05\n"
                "**  q < 0.01\n"
                "*** q < 0.001",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

    # Apply tight layout only when this function created the figure
    if created_new_axes:
        fig.tight_layout()

    return fig, ax, corr_long
