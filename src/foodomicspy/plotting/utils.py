"""
Figure-saving utilities for FoodomicsPy.
"""

from pathlib import Path


def save_figure(
    fig,
    path,
    width=1200,
    height=700,
    scale=2,
):
    """
    Save a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to save.

    path : str or pathlib.Path
        Output filename. The file extension determines
        the output format (.html, .png, .svg, .pdf).

    width : int, default=1200
        Width in pixels for static images.

    height : int, default=700
        Height in pixels for static images.

    scale : float, default=2
        Image scaling factor for static images.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.

    Examples
    --------
    >>> f.plotting.save_figure(fig, "figure.html")
    >>> f.plotting.save_figure(fig, "figure.png")
    >>> f.plotting.save_figure(fig, "figure.svg")
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = path.suffix.lower()

    if suffix == ".html":
        fig.write_html(path)

    elif suffix in {
        ".png",
        ".svg",
        ".pdf",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        fig.write_image(
            path,
            width=width,
            height=height,
            scale=scale,
        )

    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'.\n"
            "Supported formats are:\n"
            ".html, .png, .svg, .pdf, .jpg, .jpeg, .webp"
        )

    return path
