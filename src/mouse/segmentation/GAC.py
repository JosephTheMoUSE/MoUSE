"""Module implementing GAC USVs search."""
from functools import partial
from typing import Callable, List, Optional
import copy

import numpy as np
from skimage import segmentation

from mouse.utils import data_util
from mouse.utils import sound_util


def find_USVs(spec: sound_util.SpectrogramData,
              min_side_length=1,
              filter=True,
              preprocessing_fn: Optional[Callable] = None,
              level_set_fn: Optional[Callable] = None,
              **kwargs) -> List[data_util.SqueakBox]:
    """Find USVs on spectrogram `spec` using GAC.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        A spectrogram that is searched for USVs.
    min_side_length : int
        A parameter passed to `mouse.segmentation.find_bounding_boxes`.
    filter : bool
        Tells whether or not to filter USVs' bounding boxes. The filtering
        uses `mouse.data_util.filter_boxes`.
    preprocessing_fn : Callable
        Function for preprocessing the spectrogram `spec`.
    level_set_fn : Callable
        Function for generating initial level set based on preprocessed `spec`.
    kwargs
        kwargs are passed to `morphological_geodesic_active_contour` from
        `skimage.segmentation`.

    Returns
    -------
    List[data_util.SqueakBox]
        List of detected USVs' bounding boxes.
    """
    _kwargs = copy.deepcopy(kwargs)

    if level_set_fn is None:

        def level_set_fn(spectrogram: np.ndarray):
            return np.ones(spectrogram.shape, dtype=np.int8)

    if preprocessing_fn is None:
        preprocessing_fn = partial(segmentation.inverse_gaussian_gradient,
                                   sigma=5,
                                   alpha=100)
    for arg, val in [
        ("iterations", 230),
        ("smoothing", 0),
        ("threshold", 0.9),
        ("balloon", -1),
    ]:
        if arg not in _kwargs:
            _kwargs[arg] = val

    _spec = preprocessing_fn(spec.spec.numpy())

    init_level_set = level_set_fn(_spec)

    level_set = segmentation.morphological_geodesic_active_contour(
        _spec, init_level_set=init_level_set, **_kwargs)

    boxes = data_util.find_bounding_boxes(level_set, min_side_length=min_side_length)

    if filter:
        return data_util.filter_boxes(spec, boxes)
    else:
        return boxes
