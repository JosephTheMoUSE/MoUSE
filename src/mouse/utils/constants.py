"""Module containing mainly paths to data folders. Used for method development."""

from pathlib import Path

DATA_PATH = Path("/", "data", "gryzonie")
DATA_VPA = DATA_PATH.joinpath("dla_UJ2")
DATA_2018_1 = DATA_PATH.joinpath("folder1")
DATA_2018_2 = DATA_PATH.joinpath("folder2")
DATA_2020 = DATA_PATH.joinpath("LIT_DP_20201202")

SOURCES_LABELED = [DATA_VPA, DATA_2018_1, DATA_2018_2]
SOURCES_UNLABELED = [DATA_2020]  # todo: add other sources

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
