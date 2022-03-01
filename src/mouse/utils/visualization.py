"""Module containing utilities for sound visualization."""

from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colorbar
from matplotlib import rcParams
import numpy as np

from mouse.utils import data_util
from mouse.utils import sound_util


def draw_spectrogram(spec: sound_util.SpectrogramData,
                     ax: plt.Axes,
                     time_format: str = "{:.2f}",
                     freq_format: str = "{:.1f}",
                     xticks_limit: int = 10,
                     yticks_limit: int = 3,
                     cmap=rcParams["image.cmap"],
                     colormesh=None,
                     **kwargs):
    """Draws spectrogram `spec` on axes `ax`.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        The spectrogram to draw.
    ax : plt.Axes
        Ax to draw the `spec` on.
    time_format : str
        Format string for time values, defaults to "{:.2f}"
    freq_format : str
        Format string for frequency values, defaults to "{:.1f}".
    xticks_limit : int
        Maximum number of major ticks on x-axis.
    yticks_limit : int
        Maximum number of major ticks on y-axis.
    cmap : matplotlib.colors.Colormap
        Colormap used for spectrogram and colorbar. Defaults to plt.imshow
        default cmap.
    colormesh : matplotlib.collections.QuadMesh
        Object returned by plt.pcolormesh, used for optimizing computations,
        by reusing same mesh with different data
    """

    if colormesh is None:
        colormesh = ax.pcolormesh(spec.spec, cmap=cmap, **kwargs)
        ax_colorbar = ax.inset_axes([1.01, 0.05, .02, 0.95])
        colorbar.Colorbar(ax=ax_colorbar,
                          mappable=colormesh,
                          orientation="vertical")
    else:
        colormesh.set_array(spec.spec.ravel())
        # TODO: check if displayed data is correct without updating colorbar

    def get_array_formatter(array: np.array, format_str: str):
        """Create formatter for axis' labels."""

        def formatter(idx, _):
            """Return labels for axes.

            Labels are created based on index of `array` and `format_str`
            """
            if 0 <= idx < array.shape[0]:
                return format_str.format(array[int(idx)])
            return ''

        return formatter

    ax.xaxis.set_major_formatter(get_array_formatter(spec.times, time_format))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(xticks_limit))
    ax.set_xlabel("Time [s]")

    ax.yaxis.set_major_formatter(
        get_array_formatter(spec.freqs / 1000, freq_format))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(yticks_limit))
    ax.set_ylabel("Frequency [kHz]")
    return colormesh


def draw_boxes(boxes: List[data_util.SqueakBox],
               spec_height: int,
               ax: plt.Axes,
               linewidth=1,
               edgecolor='r',
               facecolor='none',
               **kwargs) -> List[patches.Rectangle]:
    """Draw bounding boxes `boxes` on axes `ax`.

    Parameters
    ----------
    boxes : List[data_util.SqueakBox]
        USVs' bounding boxes.
    spec_height : int
        The height of the spectrogram associated with boxes `boxes`.
    ax : plt.Axes
        Ax to draw the boxes on.
    linewidth
        A parameter passed to `patches.Rectangle`.
    edgecolor
        A parameter passed to `patches.Rectangle`.
    facecolor
        A parameter passed to `patches.Rectangle`.
    kwargs
        kwargs are passed to `patches.Rectangle`.
    """
    rectangles = []
    for box in boxes:
        # account for vertical flip from `draw_spectrogram`
        height = box.freq_end - box.freq_start
        width = box.t_end - box.t_start
        corner = (box.t_start, box.freq_end - height)
        rect = patches.Rectangle(corner,
                                 width,
                                 height,
                                 linewidth=linewidth,
                                 edgecolor=edgecolor,
                                 facecolor=facecolor,
                                 **kwargs)
        rectangles.append(rect)

        # Add the patch to the Axes
        ax.add_patch(rect)
    return rectangles
