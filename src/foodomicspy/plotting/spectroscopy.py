"""
Interactive spectroscopy plots for FoodomicsPy.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.colors import qualitative


def plot_nmr_interactive(
    df,
    metadata=None,
    color_by=None,
    color_map=None,
    title="NMR spectra",
    x_label="Chemical shift (ppm)",
    y_label="Intensity",
    reverse_x=True,
    renderer="browser",
    line_width=1,
    opacity=1,
    max_samples=None,
    ppm_range=None,
    downsample=1,
    show_legend=True,
    width=2500,
    height=1100,
):
    """
    Plot NMR spectra interactively using Plotly.

    Duplicate sample IDs are supported. Spectra are processed by row
    position rather than by index label.

    Parameters
    ----------
    df : pandas.DataFrame
        NMR spectral data with samples in rows and chemical-shift
        values in columns.

    metadata : pandas.DataFrame, optional
        Sample metadata. When duplicate sample IDs are present,
        metadata must contain the same number of rows as `df` and must
        be arranged in the same row order.

    color_by : str, optional
        Metadata column used to color the spectra.

    color_map : dict, optional
        Mapping between group names and colors.

    title : str, default="NMR spectra"
        Plot title.

    x_label : str, default="Chemical shift (ppm)"
        X-axis label.

    y_label : str, default="Intensity"
        Y-axis label.

    reverse_x : bool, default=True
        Reverse the chemical-shift axis.

    renderer : str or None, default="browser"
        Plotly renderer.

    line_width : float, default=1
        Spectral line width.

    opacity : float, default=1
        Line opacity.

    max_samples : int, optional
        Plot only the first N samples.

    ppm_range : tuple, optional
        Chemical-shift region to retain.

    downsample : int, default=1
        Plot every nth spectral point.

    show_legend : bool, default=True
        Show or hide the legend.

    width : int, default=2500
        Figure width in pixels.

    height : int, default=1100
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "df is empty."
        )

    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError(
            "downsample must be an integer greater than or equal to 1."
        )

    if max_samples is not None:
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError(
                "max_samples must be a positive integer or None."
            )

    if not 0 <= opacity <= 1:
        raise ValueError(
            "opacity must be between 0 and 1."
        )

    # ------------------------------------------------------------------
    # Prepare spectral data
    # ------------------------------------------------------------------
    df_plot = df.copy()

    try:
        df_plot.columns = df_plot.columns.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "All columns of df must represent numeric "
            "chemical-shift values."
        ) from error

    df_plot = df_plot.sort_index(
        axis=1,
        ascending=False,
    )

    if ppm_range is not None:
        if len(ppm_range) != 2:
            raise ValueError(
                "ppm_range must contain exactly two values."
            )

        high_ppm = max(ppm_range)
        low_ppm = min(ppm_range)

        mask = (
            (df_plot.columns >= low_ppm)
            & (df_plot.columns <= high_ppm)
        )

        df_plot = df_plot.loc[:, mask]

        if df_plot.shape[1] == 0:
            raise ValueError(
                "No chemical-shift values were found "
                "inside ppm_range."
            )

    if max_samples is not None:
        df_plot = df_plot.iloc[:max_samples]

    df_plot = df_plot.iloc[:, ::downsample]

    # ------------------------------------------------------------------
    # Prepare grouping information
    # ------------------------------------------------------------------
    group_values = None

    if color_by is not None:
        if metadata is None:
            raise ValueError(
                "metadata must be provided when color_by is used."
            )

        if not isinstance(metadata, pd.DataFrame):
            raise TypeError(
                "metadata must be a pandas DataFrame."
            )

        if color_by not in metadata.columns:
            raise KeyError(
                f"'{color_by}' was not found in metadata. "
                f"Available columns: {list(metadata.columns)}"
            )

        if len(metadata) != len(df):
            raise ValueError(
                "When duplicate sample IDs are present, metadata "
                "and df must contain the same number of rows and "
                "be arranged in the same order."
            )

        metadata_plot = metadata.copy()

        if max_samples is not None:
            metadata_plot = metadata_plot.iloc[:max_samples]

        group_values = (
            metadata_plot[color_by]
            .fillna("Unknown")
            .astype(str)
            .reset_index(drop=True)
        )

        groups = list(
            pd.unique(group_values)
        )

        if color_map is None:
            available_colors = (
                qualitative.Plotly
                + qualitative.Safe
                + qualitative.Dark24
            )

            color_map = {
                group: available_colors[
                    i % len(available_colors)
                ]
                for i, group in enumerate(groups)
            }

        else:
            missing_colors = [
                group
                for group in groups
                if group not in color_map
            ]

            if missing_colors:
                raise ValueError(
                    "color_map does not contain colors for: "
                    f"{missing_colors}"
                )

    # ------------------------------------------------------------------
    # Create interactive plot
    # ------------------------------------------------------------------
    if renderer is not None:
        pio.renderers.default = renderer

    fig = go.Figure()
    groups_added_to_legend = set()

    for row_position, (sample_id, spectrum) in enumerate(
        df_plot.iterrows()
    ):
        if color_by is None:
            trace_name = str(sample_id)
            legend_group = str(sample_id)
            trace_color = None
            display_in_legend = show_legend

            customdata = np.full(
                df_plot.shape[1],
                str(sample_id),
                dtype=object,
            )

            hovertemplate = (
                "Sample: %{customdata}<br>"
                "ppm: %{x:.4f}<br>"
                "Intensity: %{y:.3g}"
                "<extra></extra>"
            )

        else:
            group = str(
                group_values.iloc[row_position]
            )

            trace_name = group
            legend_group = group
            trace_color = color_map[group]

            display_in_legend = (
                show_legend
                and group not in groups_added_to_legend
            )

            groups_added_to_legend.add(group)

            customdata = np.full(
                df_plot.shape[1],
                str(sample_id),
                dtype=object,
            )

            hovertemplate = (
                "Sample: %{customdata}<br>"
                f"{color_by}: {group}<br>"
                "ppm: %{x:.4f}<br>"
                "Intensity: %{y:.3g}"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=df_plot.columns.to_numpy(dtype=float),
                y=spectrum.to_numpy(dtype=float),
                mode="lines",
                name=trace_name,
                legendgroup=legend_group,
                showlegend=display_in_legend,
                customdata=customdata,
                line={
                    "width": line_width,
                    "color": trace_color,
                },
                opacity=opacity,
                hovertemplate=hovertemplate,
            )
        )

    # ------------------------------------------------------------------
    # Format figure
    # ------------------------------------------------------------------
    legend_title = (
        color_by
        if color_by is not None
        else "Sample ID"
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="simple_white",
        hovermode="closest",
        showlegend=show_legend,
        legend_title_text=legend_title,
        legend={
            "groupclick": "togglegroup",
        },
        width=width,
        height=height,
        font={
            "family": "Arial",
            "size": 16,
        },
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        autorange="reversed" if reverse_x else True,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )

    return fig

def plot_ftir_interactive(
    df,
    metadata=None,
    color_by=None,
    color_map=None,
    title="FTIR spectra",
    x_label="Wavenumber (cm⁻¹)",
    y_label="Absorbance",
    reverse_x=True,
    renderer="browser",
    line_width=1,
    opacity=1,
    max_samples=None,
    wavenumber_range=None,
    downsample=1,
    show_legend=True,
):
    """
    Plot FTIR spectra interactively.

    Parameters
    ----------
    df : pandas.DataFrame
        FTIR spectra with samples in rows and wavenumbers
        in columns.

    metadata : pandas.DataFrame, optional
        Sample metadata. The index must match the sample IDs in `df`.

    color_by : str, optional
        Metadata column used to color the spectra.
        For example, color_by="sample_type".

    color_map : dict, optional
        Custom mapping between group names and colors.

    title : str, default="FTIR spectra"
        Figure title.

    x_label : str, default="Wavenumber (cm⁻¹)"
        X-axis label.

    y_label : str, default="Absorbance"
        Y-axis label.

    reverse_x : bool, default=True
        Display high wavenumbers on the left.

    renderer : str or None, default="browser"
        Plotly renderer. Use "browser" when working in Spyder.

    line_width : float, default=1
        Width of spectral lines.

    opacity : float, default=1
        Line opacity between 0 and 1.

    max_samples : int, optional
        Plot only the first N samples.

    wavenumber_range : tuple, optional
        Wavenumber region to display.

        Example:
        wavenumber_range=(4000, 600)

    downsample : int, default=1
        Plot every nth spectral point.

    show_legend : bool, default=True
        Show or hide the legend.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive FTIR figure.

    Notes
    -----
    The input data are assumed to be cleaned and standardized.
    This function does not modify the original DataFrame.

    Usage
    -----
    Plot all spectra:

    >>> fig = f.plotting.plot_ftir_interactive(df_ftir)
    >>> fig.show()

    Color spectra by sample type:

    >>> fig = f.plotting.plot_ftir_interactive(
    ...     df_ftir,
    ...     metadata=df_meta,
    ...     color_by="sample_type",
    ... )
    >>> fig.show()
    """

    df_plot = df.copy()

    # Convert the spectral axis to numeric values
    df_plot.columns = df_plot.columns.astype(float)

    # Select a wavenumber region
    if wavenumber_range is not None:
        low, high = sorted(wavenumber_range)

        mask = (
            (df_plot.columns >= low)
            & (df_plot.columns <= high)
        )

        df_plot = df_plot.loc[:, mask]

        if df_plot.shape[1] == 0:
            raise ValueError(
                "No columns were found inside wavenumber_range."
            )

    # Select a limited number of samples
    if max_samples is not None:
        df_plot = df_plot.iloc[:max_samples]

    # Reduce the number of plotted points
    if downsample > 1:
        df_plot = df_plot.iloc[:, ::downsample]

    if renderer is not None:
        pio.renderers.default = renderer

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Plot without metadata grouping
    # ------------------------------------------------------------------
    if color_by is None:
        for sample_id in df_plot.index:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.columns,
                    y=df_plot.loc[sample_id].to_numpy(),
                    mode="lines",
                    name=str(sample_id),
                    line=dict(width=line_width),
                    opacity=opacity,
                    hovertemplate=(
                        "Sample: %{fullData.name}<br>"
                        "Wavenumber: %{x:.2f} cm⁻¹<br>"
                        "Intensity: %{y:.3g}"
                        "<extra></extra>"
                    ),
                )
            )

        legend_title = "Sample ID"

    # ------------------------------------------------------------------
    # Plot using metadata grouping
    # ------------------------------------------------------------------
    else:
        if metadata is None:
            raise ValueError(
                "metadata must be provided when color_by is used."
            )

        if color_by not in metadata.columns:
            raise KeyError(
                f"'{color_by}' was not found in metadata."
            )

        missing_samples = df_plot.index.difference(metadata.index)

        if len(missing_samples) > 0:
            raise ValueError(
                "Some samples in df are missing from metadata: "
                f"{missing_samples.tolist()}"
            )

        groups = (
            metadata.loc[df_plot.index, color_by]
            .fillna("Unknown")
            .astype(str)
        )

        unique_groups = groups.unique()

        # Generate colors automatically
        if color_map is None:
            colors = (
                qualitative.Plotly
                + qualitative.Safe
                + qualitative.Dark24
            )

            color_map = {
                group: colors[i % len(colors)]
                for i, group in enumerate(unique_groups)
            }

        legend_groups_added = set()

        for sample_id in df_plot.index:
            group = groups.loc[sample_id]

            fig.add_trace(
                go.Scatter(
                    x=df_plot.columns,
                    y=df_plot.loc[sample_id].to_numpy(),
                    mode="lines",
                    name=group,
                    legendgroup=group,
                    showlegend=(
                        show_legend
                        and group not in legend_groups_added
                    ),
                    line=dict(
                        width=line_width,
                        color=color_map[group],
                    ),
                    opacity=opacity,
                    customdata=[str(sample_id)] * df_plot.shape[1],
                    hovertemplate=(
                        "Sample: %{customdata}<br>"
                        f"{color_by}: {group}<br>"
                        "Wavenumber: %{x:.2f} cm⁻¹<br>"
                        "Intensity: %{y:.3g}"
                        "<extra></extra>"
                    ),
                )
            )

            legend_groups_added.add(group)

        legend_title = color_by

    # ------------------------------------------------------------------
    # Format the figure
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="simple_white",
        hovermode="closest",
        showlegend=show_legend,
        legend_title_text=legend_title,
        legend=dict(
            groupclick="togglegroup",
        ),
        width=2500,
        height=1100,
        font=dict(
            family="Arial",
            size=16,
        ),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        autorange="reversed" if reverse_x else True,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )

    return fig
