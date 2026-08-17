# -*- coding: utf-8 -*-
"""
PLS plotting functions for FoodomicsPy.

The module provides diagnostic and interpretation plots for results
returned by foodomicspy.chemometrics.perform_pls().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _validate_pls_result(result, required_keys=None):
    """
    Validate a PLS result dictionary.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "'result' must be the dictionary returned by perform_pls()."
        )

    if required_keys is None:
        required_keys = []

    missing = set(required_keys).difference(result)

    if missing:
        raise KeyError(
            "The PLS result is missing the following required entries: "
            f"{sorted(missing)}"
        )


def _prepare_metadata(result, metadata=None, color_by=None):
    """
    Align optional metadata to samples stored in a PLS result.

    Returns
    -------
    pd.Series or None
        Metadata variable aligned to result sample order.
    """

    if color_by is None:
        return None

    if metadata is None:
        raise ValueError(
            "'metadata' must be supplied when 'color_by' is used."
        )

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "'metadata' must be a pandas DataFrame."
        )

    if color_by not in metadata.columns:
        raise KeyError(
            f"'{color_by}' was not found in metadata."
        )

    sample_names = pd.Index(
        result["sample_names"]
    )

    missing_samples = sample_names.difference(
        metadata.index
    )

    if len(missing_samples) > 0:
        raise ValueError(
            "Some samples in the PLS model are missing from metadata: "
            f"{list(missing_samples[:10])}"
        )

    values = metadata.loc[
        sample_names,
        color_by,
    ].copy()

    return values


def _save_figure(fig, save_path=None, dpi=600):
    """
    Save a matplotlib figure when requested.
    """

    if save_path is None:
        return

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if save_path.suffix == "":
        save_path = save_path.with_suffix(
            ".png"
        )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )


def _style_axis(ax):
    """
    Apply common FoodomicsPy axis styling.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        True,
        linewidth=0.6,
        alpha=0.20,
    )

    ax.set_axisbelow(True)


def _scatter_with_metadata(
    ax,
    x,
    y,
    metadata_values=None,
    point_size=55,
    point_alpha=0.85,
    marker="o",
):
    """
    Create a scatter plot with optional metadata coloring.

    Categorical metadata produces one color per category and a legend.
    Continuous numerical metadata produces a continuous colormap.

    Returns
    -------
    scatter_objects : list
        Matplotlib scatter artists.

    continuous_scatter : PathCollection or None
        Scatter object used for a colorbar when the metadata variable
        is continuous.
    """

    scatter_objects = []
    continuous_scatter = None

    # -------------------------------------------------------------------------
    # No metadata coloring
    # -------------------------------------------------------------------------
    if metadata_values is None:

        scatter = ax.scatter(
            x,
            y,
            s=point_size,
            alpha=point_alpha,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        scatter_objects.append(
            scatter
        )

        return scatter_objects, continuous_scatter

    # -------------------------------------------------------------------------
    # Continuous numerical metadata
    # -------------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(
        metadata_values
    ):

        continuous_scatter = ax.scatter(
            x,
            y,
            c=metadata_values.to_numpy(),
            s=point_size,
            alpha=point_alpha,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        scatter_objects.append(
            continuous_scatter
        )

        return scatter_objects, continuous_scatter

    # -------------------------------------------------------------------------
    # Categorical metadata
    # -------------------------------------------------------------------------
    categories = pd.unique(
        metadata_values.astype(str)
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    for i, category in enumerate(categories):

        mask = (
            metadata_values.astype(str)
            == category
        ).to_numpy()

        scatter = ax.scatter(
            np.asarray(x)[mask],
            np.asarray(y)[mask],
            s=point_size,
            alpha=point_alpha,
            edgecolor="white",
            linewidth=0.8,
            marker=marker,
            color=cmap(i % cmap.N),
            label=str(category),
            zorder=3,
        )

        scatter_objects.append(
            scatter
        )

    return scatter_objects, continuous_scatter


# =============================================================================
# PLS COMPONENT OPTIMIZATION
# =============================================================================

def plot_pls_components(
    result,
    ax=None,
    title="Model complexity",
    show_selected=True,
    save_path=None,
    dpi=600,
):
    """
    Plot cross-validation error against number of PLS components.

    Parameters
    ----------
    result : dict
        Result returned by perform_pls().

    ax : matplotlib.axes.Axes or None
        Existing axes. When None, a new figure is created.

    title : str or None
        Plot title.

    show_selected : bool, default=True
        Highlight the selected number of components.

    save_path : str, pathlib.Path, or None
        Optional file path.

    dpi : int, default=600
        Resolution for raster output.

    Returns
    -------
    fig, ax
    """

    _validate_pls_result(
        result,
        required_keys=[
            "component_results",
            "n_components",
        ],
    )

    component_results = result[
        "component_results"
    ]

    if not {
        "n_components",
        "rmsecv",
    }.issubset(component_results.columns):

        raise KeyError(
            "component_results must contain "
            "'n_components' and 'rmsecv'."
        )

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(
            figsize=(6.2, 4.5)
        )
    else:
        fig = ax.figure

    x = component_results[
        "n_components"
    ].to_numpy()

    y = component_results[
        "rmsecv"
    ].to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.8,
        markersize=5,
    )

    best_lv = int(
        result["n_components"]
    )

    best_row = component_results.loc[
        component_results["n_components"]
        == best_lv
    ]

    if not best_row.empty:

        best_rmsecv = float(
            best_row["rmsecv"].iloc[0]
        )

        if show_selected:

            ax.scatter(
                best_lv,
                best_rmsecv,
                s=90,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

            ax.axvline(
                best_lv,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )

            ax.annotate(
                (
                    f"Selected = {best_lv} LV"
                    f"\nRMSECV = {best_rmsecv:.3f}"
                ),
                xy=(
                    best_lv,
                    best_rmsecv,
                ),
                xytext=(8, 15),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel(
        "Number of latent variables"
    )

    ax.set_ylabel(
        "RMSECV"
    )

    if title is not None:
        ax.set_title(
            title,
            loc="left",
        )

    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

    _style_axis(ax)

    if created_figure:
        fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, ax


# =============================================================================
# MEASURED VS PREDICTED
# =============================================================================

def plot_pls_predictions(
    result,
    metadata=None,
    color_by=None,
    prediction="cv",
    ax=None,
    point_size=55,
    point_alpha=0.85,
    show_identity_line=True,
    show_regression_line=False,
    show_metrics=True,
    show_sample_labels=False,
    label_residual_threshold=None,
    title="Prediction performance",
    save_path=None,
    dpi=600,
):
    """
    Plot measured versus predicted response values.

    Parameters
    ----------
    result : dict
        Result returned by perform_pls().

    metadata : pandas.DataFrame or None
        Optional sample metadata.

        Its index must contain the sample IDs used in the PLS model.

    color_by : str or None
        Metadata column used to color observations.

        Categorical columns generate a legend.

        Numerical columns generate a continuous color scale.

    prediction : {"cv", "calibration"}, default="cv"
        Prediction type to display.

    ax : matplotlib.axes.Axes or None
        Existing axes.

    point_size : float, default=55
        Scatter point size.

    point_alpha : float, default=0.85
        Point transparency.

    show_identity_line : bool, default=True
        Draw the y=x line.

    show_regression_line : bool, default=False
        Draw a regression line between measured and predicted values.

    show_metrics : bool, default=True
        Display major model statistics.

    show_sample_labels : bool, default=False
        Label every sample.

    label_residual_threshold : float or None
        Label samples with absolute residual >= threshold.

    title : str or None
        Plot title.

    save_path : str, pathlib.Path, or None
        Optional output file.

    Returns
    -------
    fig, ax
    """

    _validate_pls_result(
        result,
        required_keys=[
            "y_measured",
            "y_cv_predicted",
            "y_calibration_predicted",
            "sample_names",
        ],
    )

    if prediction not in {
        "cv",
        "calibration",
    }:
        raise ValueError(
            "'prediction' must be 'cv' or 'calibration'."
        )

    y_measured = np.asarray(
        result["y_measured"]
    ).reshape(-1)

    if prediction == "cv":

        y_predicted = np.asarray(
            result["y_cv_predicted"]
        ).reshape(-1)

        residuals = (
            y_measured
            - y_predicted
        )

    else:

        y_predicted = np.asarray(
            result["y_calibration_predicted"]
        ).reshape(-1)

        residuals = (
            y_measured
            - y_predicted
        )

    metadata_values = _prepare_metadata(
        result,
        metadata=metadata,
        color_by=color_by,
    )

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(
            figsize=(5.7, 5.3)
        )
    else:
        fig = ax.figure

    # -------------------------------------------------------------------------
    # Axis limits
    # -------------------------------------------------------------------------
    all_values = np.concatenate(
        [
            y_measured,
            y_predicted,
        ]
    )

    value_min = np.nanmin(
        all_values
    )

    value_max = np.nanmax(
        all_values
    )

    value_range = (
        value_max
        - value_min
    )

    if np.isclose(
        value_range,
        0,
    ):
        value_range = 1

    padding = (
        value_range * 0.07
    )

    axis_min = (
        value_min - padding
    )

    axis_max = (
        value_max + padding
    )

    # -------------------------------------------------------------------------
    # Identity line
    # -------------------------------------------------------------------------
    if show_identity_line:

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            linewidth=1.2,
            color="black",
            label="Identity line",
            zorder=1,
        )

    # -------------------------------------------------------------------------
    # Scatter
    # -------------------------------------------------------------------------
    _, continuous_scatter = (
        _scatter_with_metadata(
            ax=ax,
            x=y_measured,
            y=y_predicted,
            metadata_values=metadata_values,
            point_size=point_size,
            point_alpha=point_alpha,
        )
    )

    # -------------------------------------------------------------------------
    # Regression line
    # -------------------------------------------------------------------------
    if show_regression_line:

        slope, intercept = np.polyfit(
            y_measured,
            y_predicted,
            deg=1,
        )

        regression_x = np.array(
            [
                axis_min,
                axis_max,
            ]
        )

        regression_y = (
            slope * regression_x
            + intercept
        )

        ax.plot(
            regression_x,
            regression_y,
            linewidth=1.5,
            label="Regression line",
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    if show_metrics:

        if prediction == "cv":

            metric_lines = [
                f"LV = {result['n_components']}",
                f"RMSECV = {result['rmsecv']:.3f}",
                rf"$R^2_{{CV}}$ = {result['r2cv']:.3f}",
            ]

            if "q2cv" in result:
                metric_lines.append(
                    rf"$Q^2$ = {result['q2cv']:.3f}"
                )

        else:

            metric_lines = [
                f"LV = {result['n_components']}",
                f"RMSEC = {result['rmsec']:.3f}",
                rf"$R^2_C$ = {result['r2c']:.3f}",
            ]

        ax.text(
            0.05,
            0.95,
            "\n".join(
                metric_lines
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.85",
                "alpha": 0.9,
            },
        )

    # -------------------------------------------------------------------------
    # Sample labels
    # -------------------------------------------------------------------------
    sample_names = np.asarray(
        result["sample_names"]
    )

    if show_sample_labels:

        label_mask = np.ones(
            len(sample_names),
            dtype=bool,
        )

    elif label_residual_threshold is not None:

        if label_residual_threshold < 0:
            raise ValueError(
                "'label_residual_threshold' must be >= 0."
            )

        label_mask = (
            np.abs(residuals)
            >= label_residual_threshold
        )

    else:

        label_mask = np.zeros(
            len(sample_names),
            dtype=bool,
        )

    for i in np.where(
        label_mask
    )[0]:

        ax.annotate(
            str(sample_names[i]),
            (
                y_measured[i],
                y_predicted[i],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    y_name = result.get(
        "y_name"
    )

    if y_name is None:
        y_name = "Response"

    ax.set_xlabel(
        f"Measured {y_name}"
    )

    if prediction == "cv":

        ax.set_ylabel(
            f"Cross-validated predicted {y_name}"
        )

    else:

        ax.set_ylabel(
            f"Calibration predicted {y_name}"
        )

    ax.set_xlim(
        axis_min,
        axis_max,
    )

    ax.set_ylim(
        axis_min,
        axis_max,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if title is not None:
        ax.set_title(
            title,
            loc="left",
        )

    # -------------------------------------------------------------------------
    # Legend / colorbar
    # -------------------------------------------------------------------------
    if metadata_values is not None:

        if pd.api.types.is_numeric_dtype(
            metadata_values
        ):

            cbar = fig.colorbar(
                continuous_scatter,
                ax=ax,
            )

            cbar.set_label(
                color_by
            )

        else:

            ax.legend(
                title=color_by,
                frameon=False,
            )

    _style_axis(ax)

    if created_figure:
        fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, ax


# =============================================================================
# RESIDUALS
# =============================================================================

def plot_pls_residuals(
    result,
    metadata=None,
    color_by=None,
    prediction="cv",
    ax=None,
    point_size=55,
    point_alpha=0.85,
    show_sd_lines=True,
    sd_threshold=2,
    show_sample_labels=False,
    label_residual_threshold=None,
    title="Residual diagnostics",
    save_path=None,
    dpi=600,
):
    """
    Plot residuals against predicted values.
    """

    _validate_pls_result(
        result,
        required_keys=[
            "y_measured",
            "y_cv_predicted",
            "y_calibration_predicted",
            "sample_names",
        ],
    )

    if prediction not in {
        "cv",
        "calibration",
    }:
        raise ValueError(
            "'prediction' must be 'cv' or 'calibration'."
        )

    y_measured = np.asarray(
        result["y_measured"]
    ).reshape(-1)

    if prediction == "cv":

        y_predicted = np.asarray(
            result["y_cv_predicted"]
        ).reshape(-1)

    else:

        y_predicted = np.asarray(
            result["y_calibration_predicted"]
        ).reshape(-1)

    residuals = (
        y_measured
        - y_predicted
    )

    metadata_values = _prepare_metadata(
        result,
        metadata=metadata,
        color_by=color_by,
    )

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(
            figsize=(6.0, 4.5)
        )
    else:
        fig = ax.figure

    # -------------------------------------------------------------------------
    # Zero line
    # -------------------------------------------------------------------------
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.1,
        color="black",
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # Scatter
    # -------------------------------------------------------------------------
    _, continuous_scatter = (
        _scatter_with_metadata(
            ax=ax,
            x=y_predicted,
            y=residuals,
            metadata_values=metadata_values,
            point_size=point_size,
            point_alpha=point_alpha,
        )
    )

    # -------------------------------------------------------------------------
    # SD limits
    # -------------------------------------------------------------------------
    if show_sd_lines:

        residual_sd = np.std(
            residuals,
            ddof=1,
        )

        if (
            np.isfinite(residual_sd)
            and residual_sd > 0
        ):

            upper = (
                sd_threshold
                * residual_sd
            )

            lower = -upper

            ax.axhline(
                upper,
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )

            ax.axhline(
                lower,
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )

    # -------------------------------------------------------------------------
    # Sample labels
    # -------------------------------------------------------------------------
    sample_names = np.asarray(
        result["sample_names"]
    )

    if show_sample_labels:

        label_mask = np.ones(
            len(sample_names),
            dtype=bool,
        )

    elif label_residual_threshold is not None:

        if label_residual_threshold < 0:
            raise ValueError(
                "'label_residual_threshold' must be >= 0."
            )

        label_mask = (
            np.abs(residuals)
            >= label_residual_threshold
        )

    else:

        label_mask = np.zeros(
            len(sample_names),
            dtype=bool,
        )

    for i in np.where(
        label_mask
    )[0]:

        ax.annotate(
            str(sample_names[i]),
            (
                y_predicted[i],
                residuals[i],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    y_name = result.get(
        "y_name"
    )

    if y_name is None:
        y_name = "Response"

    if prediction == "cv":

        ax.set_xlabel(
            f"Cross-validated predicted {y_name}"
        )

    else:

        ax.set_xlabel(
            f"Calibration predicted {y_name}"
        )

    ax.set_ylabel(
        "Residual (measured − predicted)"
    )

    if title is not None:
        ax.set_title(
            title,
            loc="left",
        )

    # -------------------------------------------------------------------------
    # Legend / colorbar
    # -------------------------------------------------------------------------
    if metadata_values is not None:

        if pd.api.types.is_numeric_dtype(
            metadata_values
        ):

            cbar = fig.colorbar(
                continuous_scatter,
                ax=ax,
            )

            cbar.set_label(
                color_by
            )

        else:

            ax.legend(
                title=color_by,
                frameon=False,
            )

    _style_axis(ax)

    if created_figure:
        fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, ax


# =============================================================================
# REGRESSION COEFFICIENTS
# =============================================================================

def plot_pls_coefficients(
    result,
    ax=None,
    reverse_x=False,
    zero_line=True,
    title="PLS regression coefficients",
    save_path=None,
    dpi=600,
):
    """
    Plot PLS regression coefficients against predictor variables.
    """

    _validate_pls_result(
        result,
        required_keys=[
            "coefficients",
        ],
    )

    coefficients = result[
        "coefficients"
    ]

    if isinstance(
        coefficients,
        pd.Series,
    ):

        x = coefficients.index
        y = coefficients.to_numpy()

    else:

        y = np.asarray(
            coefficients
        ).reshape(-1)

        x = result.get(
            "feature_names",
            np.arange(len(y)),
        )

    # Try to convert spectral axis to numerical values
    try:
        x_plot = np.asarray(
            x,
            dtype=float,
        )

    except (TypeError, ValueError):
        x_plot = np.arange(
            len(y)
        )

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(
            figsize=(7.5, 4.3)
        )
    else:
        fig = ax.figure

    ax.plot(
        x_plot,
        y,
        linewidth=1.5,
    )

    if zero_line:

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.6,
        )

    ax.set_xlabel(
        "Variable"
    )

    ax.set_ylabel(
        "Regression coefficient"
    )

    if title is not None:
        ax.set_title(
            title,
            loc="left",
        )

    if reverse_x:
        ax.invert_xaxis()

    _style_axis(ax)

    if created_figure:
        fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, ax


# =============================================================================
# PLS SCORES
# =============================================================================

def plot_pls_scores(
    result,
    metadata=None,
    color_by=None,
    components=(1, 2),
    ax=None,
    point_size=60,
    point_alpha=0.85,
    show_sample_labels=False,
    title="PLS scores",
    save_path=None,
    dpi=600,
):
    """
    Plot two PLS X-score components.
    """

    _validate_pls_result(
        result,
        required_keys=[
            "x_scores",
            "sample_names",
        ],
    )

    scores = result[
        "x_scores"
    ]

    if not isinstance(
        scores,
        pd.DataFrame,
    ):

        scores = pd.DataFrame(
            scores,
            index=result[
                "sample_names"
            ],
        )

    component_x = int(
        components[0]
    )

    component_y = int(
        components[1]
    )

    if component_x < 1 or component_y < 1:
        raise ValueError(
            "PLS components are numbered starting from 1."
        )

    if (
        component_x > scores.shape[1]
        or component_y > scores.shape[1]
    ):
        raise ValueError(
            f"The model contains only {scores.shape[1]} components."
        )

    x = scores.iloc[
        :,
        component_x - 1
    ].to_numpy()

    y = scores.iloc[
        :,
        component_y - 1
    ].to_numpy()

    metadata_values = _prepare_metadata(
        result,
        metadata=metadata,
        color_by=color_by,
    )

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(
            figsize=(5.8, 5.0)
        )
    else:
        fig = ax.figure

    _, continuous_scatter = (
        _scatter_with_metadata(
            ax=ax,
            x=x,
            y=y,
            metadata_values=metadata_values,
            point_size=point_size,
            point_alpha=point_alpha,
        )
    )

    ax.axhline(
        0,
        linewidth=0.8,
        color="black",
        alpha=0.4,
    )

    ax.axvline(
        0,
        linewidth=0.8,
        color="black",
        alpha=0.4,
    )

    if show_sample_labels:

        sample_names = np.asarray(
            result["sample_names"]
        )

        for i, sample in enumerate(
            sample_names
        ):

            ax.annotate(
                str(sample),
                (
                    x[i],
                    y[i],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )

    ax.set_xlabel(
        f"LV{component_x} scores"
    )

    ax.set_ylabel(
        f"LV{component_y} scores"
    )

    if title is not None:
        ax.set_title(
            title,
            loc="left",
        )

    if metadata_values is not None:

        if pd.api.types.is_numeric_dtype(
            metadata_values
        ):

            cbar = fig.colorbar(
                continuous_scatter,
                ax=ax,
            )

            cbar.set_label(
                color_by
            )

        else:

            ax.legend(
                title=color_by,
                frameon=False,
            )

    _style_axis(ax)

    if created_figure:
        fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, ax


# =============================================================================
# PLS SUMMARY FIGURE
# =============================================================================

def plot_pls(
    result,
    metadata=None,
    color_by=None,
    figsize=(15, 4.7),
    point_size=55,
    point_alpha=0.85,
    show_regression_line=False,
    show_sample_labels=False,
    label_residual_threshold=None,
    title=None,
    save_path=None,
    dpi=600,
):
    """
    Create a three-panel PLS diagnostic summary.

    Panels
    ------
    A. RMSECV versus number of latent variables
    B. Measured versus cross-validated predicted values
    C. Cross-validated residuals versus predicted values

    Parameters
    ----------
    result : dict
        Result returned by perform_pls().

    metadata : pandas.DataFrame or None
        Optional metadata table.

    color_by : str or None
        Metadata variable used to color samples in prediction and
        residual plots.

    figsize : tuple, default=(15, 4.7)
        Overall figure size.

    point_size : float, default=55
        Scatter point size.

    point_alpha : float, default=0.85
        Scatter point transparency.

    show_regression_line : bool, default=False
        Show regression line in measured-versus-predicted panel.

    show_sample_labels : bool, default=False
        Label every sample.

    label_residual_threshold : float or None
        Label samples having absolute CV residual greater than or equal
        to this value.

    title : str or None
        Overall figure title.

    save_path : str, pathlib.Path, or None
        Optional output path.

    dpi : int, default=600
        Resolution for raster output.

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : numpy.ndarray
        Three matplotlib axes.
    """

    _validate_pls_result(
        result,
        required_keys=[
            "component_results",
            "n_components",
            "y_measured",
            "y_cv_predicted",
            "sample_names",
        ],
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
    )

    # -------------------------------------------------------------------------
    # A. Component optimization
    # -------------------------------------------------------------------------
    plot_pls_components(
        result=result,
        ax=axes[0],
        title="Model complexity",
    )

    # -------------------------------------------------------------------------
    # B. Measured vs predicted
    # -------------------------------------------------------------------------
    plot_pls_predictions(
        result=result,
        metadata=metadata,
        color_by=color_by,
        prediction="cv",
        ax=axes[1],
        point_size=point_size,
        point_alpha=point_alpha,
        show_regression_line=show_regression_line,
        show_sample_labels=show_sample_labels,
        label_residual_threshold=label_residual_threshold,
        title="Prediction performance",
    )

    # -------------------------------------------------------------------------
    # C. Residuals
    # -------------------------------------------------------------------------
    plot_pls_residuals(
        result=result,
        metadata=metadata,
        color_by=color_by,
        prediction="cv",
        ax=axes[2],
        point_size=point_size,
        point_alpha=point_alpha,
        show_sample_labels=show_sample_labels,
        label_residual_threshold=label_residual_threshold,
        title="Residual diagnostics",
    )

    # -------------------------------------------------------------------------
    # Panel labels
    # -------------------------------------------------------------------------
    for ax, panel_label in zip(
        axes,
        ["A", "B", "C"],
    ):

        ax.text(
            -0.14,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )

    # -------------------------------------------------------------------------
    # Overall title
    # -------------------------------------------------------------------------
    if title is not None:

        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()

    _save_figure(
        fig,
        save_path,
        dpi,
    )

    return fig, axes