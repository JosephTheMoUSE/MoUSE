from typing import Union, List, Optional, Callable
from mouse.utils.data_util import SqueakBox
from mouse.utils.sound_util import SpectrogramData


def classify_USVs(spec: SpectrogramData,
                  squeak_boxes: List[SqueakBox],
                  threshold: Union[int, float],
                  low_label: str = 'low freq',
                  high_label: str = 'high freq',
                  callback: Optional[Callable] = None):
    """Run simple classification of `squeak_boxes`.

    Parameters
    ----------
    spec : sound_util.SpectrogramData
        A spectrogram from which squeak_boxes come from.
    squeak_boxes : List[SqueakBox]
        List of `SqueakBoxe`s.
    threshold : Union[int, float]
        Frequency threshold that divides USVs int high and low.
    low_label : str
        Label to add to low frequency USVs.
    high_label : str
        Label to add to low frequency USVs.
    callback : Optional[Callable]
        Function called after each SqueakBox is processed.

    Returns
    -------
    List[data_util.SqueakBox]
        List of annotated USVs'.
    """
    if isinstance(threshold, float):
        threshold = spec.freq_to_pixels(threshold)
    for squeak_box in squeak_boxes:
        mean_freq = (squeak_box.freq_end + squeak_box.freq_start) / 2
        if mean_freq > threshold:
            squeak_box.label = high_label
        else:
            squeak_box.label = low_label
        if callback is not None:
            callback()

    return squeak_boxes
