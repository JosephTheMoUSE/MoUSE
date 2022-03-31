"""Module containing mainly paths to data folders. Used for method development."""
from __future__ import annotations

import json
import pathlib
from typing import List, Union

import environ

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent


def _path_converter(
        path: Union[None, str,
                    List[str]]) -> Union[None, pathlib.Path, List[pathlib.Path]]:
    if path is None:
        return None
    elif isinstance(path, List):
        return [pathlib.Path(p) for p in path]
    else:
        return pathlib.Path(path)


@environ.config
class DataSources:
    """Container for data paths.

    To create variables use:
     - `DataSources.from_json()` to config from "user_config.json"
     - `DataSources.from_environ()` to config from the environment
    """

    selected_source: pathlib.Path = environ.var(name="MAIN_SOURCE",
                                                default=None,
                                                converter=_path_converter)
    labeled_sources: List[pathlib.Path] = environ.var(
        name="LABELED_SOURCES",
        default=None,
        converter=_path_converter,
    )

    unlabeled_sources: List[pathlib.Path] = environ.var(
        name="UNLABELED_SOURCES",
        default=None,
        converter=_path_converter,
    )

    @classmethod
    def from_json(self) -> DataSources:
        """Load configuration from "user_config.json"."""
        config = PROJECT_ROOT.joinpath("user_config.json")
        with config.open("r") as fp:
            return DataSources.from_environ(json.load(fp))


# names of columns in labeled data files
COL_SELECTION = "Selection"
COL_VIEW = "View"
COL_CHANNEL = "Channel"
COL_BEGIN_TIME = "Begin_Time_(s)"
COL_END_TIME = "End_Time_(s)"
COL_DELTA_TIME = "Delta_Time_(s)"
COL_LOW_FREQ = "Low_Freq_(Hz)"
COL_HIGH_FREQ = "High_Freq_(Hz)"
COL_CENTER_FREQ = "Center_Freq_(Hz)"
COL_PEAK_FREQ = "Peak_Freq_(Hz)"
COL_BEGIN_FILE = "Begin_File"
COL_DELTA_FREQ = "Delta_Freq_(Hz)"
COL_USV_TYPE = "USV_TYPE"
