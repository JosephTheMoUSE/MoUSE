from __future__ import annotations

import argparse
import json
import pathlib
from functools import partial
from typing import Any
from typing import Dict, List

import pandas as pd
import ray
from ray import tune
from skimage import segmentation

import mouse
import mouse.segmentation
import mouse.segmentation
from mouse.denoising import denoising
from mouse.segmentation import balloon_from_latent, threshold_from_latent
from mouse.utils import data_util, sound_util, metrics
from mouse.utils.data_util import SignalNoise
from mouse.utils.metrics import Metric

# Constants
# # Paths
NOISES = pathlib.Path("noises")
RESULTS_ROOT = pathlib.Path("optimisation_results")
OPTIMISATION_FINAL_RESULTS = RESULTS_ROOT.joinpath("final")
OPTIMISATION_PARTIAL_RESULTS = RESULTS_ROOT.joinpath("partial")
DETECTIONS = RESULTS_ROOT.joinpath("detections")
METRICS = RESULTS_ROOT.joinpath("metrics")

# # Optimisation-related constants
MAX_CONCURRENT = 3  # max number of concurrent optimisation trials
TRAIN_TIME = 10.0
SAVE_METRICS = True
RANDOM_SEARCH_STEPS = 40  # number of trials configured by Bayesian optimisation
NOT_RANDOM_SEARCH_STEPS = (
    160  # number of random trials for initialising the optimisation process
)
NUM_SAMPLES = RANDOM_SEARCH_STEPS + NOT_RANDOM_SEARCH_STEPS
BETA = 100.0
ALPHA = 200

# # Spectrogram generation
N_FFT = 512
WIN_LENGTH = 256
HOP_LENGTH = 128


def load_noises(recording_name: str) -> List[SignalNoise]:
    with NOISES.joinpath(recording_name).open("r") as fp:
        noise_times: Dict[str, dict] = json.load(fp)

    return [
        SignalNoise(config_id=config_id, start=time_info["start"], end=time_info["end"])
        for config_id, time_info in noise_times.items()
    ]


def load_gac_config(recording_name: str, config_id: str):
    with OPTIMISATION_FINAL_RESULTS.joinpath(recording_name).open("r") as fp:
        configs: Dict[str, dict] = json.load(fp)

    return configs[config_id]


def is_configuration_present(
    target_folder: pathlib.Path, recording_name: str, config_id: str
) -> bool:
    result_path = target_folder.joinpath(recording_name)

    if not result_path.exists():
        return False

    with result_path.open("r") as fp:
        result = json.load(fp)

    return config_id in result


def find_noises_in_recording(data_folder: data_util.DataFolder, recording_name: str):
    """Create .json noise file listing noises from recording.

    Create noise file for recording named `recording_name`. The file is created in
    `NOISES` folder and it's named `recording_name`.

    Format of this file is as follows:
    {
        <configuration_identifier_1>: {"start": <noise start [seconds]>,
                                       "end": <noise end [seconds]>},
        <configuration_identifier_2>: {"start": <noise start [seconds]>,
                                       "end": <noise end [seconds]>},
        ...
    }
    """
    raise NotImplementedError("Noise finding is not yet implemented.")


def prepare_spec(
    squeak_signal: sound_util.SignalData, noise: SignalNoise
) -> sound_util.SpectrogramData:
    spec: sound_util.SpectrogramData = sound_util.signal_spectrogram(
        squeak_signal,
        start=0.0,
        end=1.0,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )
    noise_spec = sound_util.clip_spectrogram(
        spec=spec, t_start=noise.start, t_end=noise.end
    )
    denoising.noise_gate_filter(
        spectrogram=spec,
        noise_spectrogram=noise_spec,
        n_grad_freq=3,
        n_grad_time=3,
        n_std_thresh=1.0,
        noise_decrease=0.5,
    )
    return spec


class _OptimisationCallback(tune.Callback):
    """Save partial optimisation results to a file."""

    def __init__(
        self,
        recording_name: str,
        config_id: str,
        beta: float,
        training_time: float,
        train_boxes: int,
    ):
        self.config_id = config_id
        self.results = OPTIMISATION_PARTIAL_RESULTS.joinpath(recording_name)
        self.beta = beta
        self.training_time = training_time
        self.train_boxes = train_boxes
        self.results_df: pd.DataFrame = pd.DataFrame()
        if self.results.exists():
            self.results_df = pd.read_csv(self.results)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        new_row = dict()

        for key in ["score", "precision", "recall", "box_count"]:
            new_row[key] = result[key]

        new_row["config_id"] = self.config_id
        new_row["train_boxes"] = self.train_boxes
        new_row["beta"] = self.beta
        new_row["training_time"] = self.training_time
        new_row["balloon"] = round(
            balloon_from_latent(result["config"]["_balloon_latent"]), 3
        )
        new_row["threshold"] = round(
            threshold_from_latent(result["config"]["_balloon_latent"]), 3
        )
        new_row["sigma"] = round(result["config"]["sigma"], 3)
        new_row["iters"] = int(result["config"]["iterations"])
        new_row["flood_threshold"] = round(result["config"]["flood_threshold"], 3)
        new_row["smoothing"] = int(result["config"]["smoothing"])

        self.results_df = self.results_df.append([new_row])
        self.results_df.to_csv(self.results, index=False)


def save_best_configuration(
    analysis: tune.ExperimentAnalysis,
    train_boxes_count: int,
    config_id: str,
    beta: float,
    recording_name: str,
    time_end: float,
):
    balloon = balloon_from_latent(analysis.best_config["_balloon_latent"])
    threshold = threshold_from_latent(analysis.best_config["_balloon_latent"])
    result_path = OPTIMISATION_FINAL_RESULTS.joinpath(recording_name)

    result = dict()
    if result_path.exists():
        with result_path.open("r") as fp:
            result = json.load(fp)

    config_results: Dict[str, Any] = dict()

    config_results["train_boxes"] = train_boxes_count
    config_results["beta"] = beta
    config_results["training_time"] = time_end
    config_results[f"balloon"] = balloon
    config_results[f"threshold"] = threshold
    config_results.update(analysis.best_config)

    result[config_id] = config_results

    with result_path.open("w") as fp:
        json.dump(result, fp, indent=0)


def optimise(
    beta,
    ground_truth,
    config_id,
    num_samples,
    random_search_steps,
    recording_name,
    spec,
    train_time,
):
    spec_train, ground_truth_train = data_util.clip_spec_and_boxes(
        spec=spec, t_end=train_time, boxes=ground_truth
    )
    print(f"Number of training USVs: {len(ground_truth_train)}")
    analysis = mouse.segmentation.optimisation.optimise_gac(
        spec=spec_train,
        ground_truth=ground_truth_train,
        metric=Metric.F_BETA,
        threshold_metric=0.1,
        num_samples=num_samples,
        random_search_steps=random_search_steps,
        max_concurrent=MAX_CONCURRENT,
        alpha=ALPHA,
        beta=beta,
        callbacks=[
            _OptimisationCallback(
                recording_name=recording_name,
                beta=beta,
                training_time=train_time,
                config_id=config_id,
                train_boxes=len(ground_truth_train),
            )
        ],
        remove_ray_results=False,
    )
    ray.shutdown()
    save_best_configuration(
        analysis=analysis,
        beta=beta,
        recording_name=recording_name,
        config_id=config_id,
        time_end=train_time,
        train_boxes_count=len(ground_truth_train),
    )


def detect_and_process_detections(
    ground_truth: List[data_util.SqueakBox],
    config_id: str,
    recording_name: str,
    spec: sound_util.SpectrogramData,
    train_time: float,
    save_metrics: bool,
    beta: float,
):
    config = load_gac_config(recording_name=recording_name, config_id=config_id)
    level_set = mouse.segmentation.eroded_level_set_generator(
        round(config["flood_threshold"], 3)
    )

    detections = mouse.segmentation.find_USVs(
        spec=spec,
        balloon=config["balloon"],
        iterations=int(config["iterations"]),
        threshold=config["threshold"],
        smoothing=int(config["smoothing"]),
        min_side_length=1,
        filter=True,
        level_set=level_set,
        preprocessing_fn=partial(
            segmentation.inverse_gaussian_gradient, sigma=config["sigma"], alpha=ALPHA
        ),
    )
    result = dict()
    result_path = DETECTIONS.joinpath(recording_name)
    if result_path.exists():
        with result_path.open("r") as fp:
            result = json.load(fp)

    result[config_id] = [
        data_util.SqueakBoxSI.from_squeak_box(squeak_box=box, spec_data=spec).to_dict()
        for box in detections
    ]

    with result_path.open("w") as fp:
        json.dump(result, fp, indent=0)

    if save_metrics:
        print(f"Calculating metrics")
        spec_test, test_gt = data_util.clip_spec_and_boxes(
            spec=spec, t_start=train_time, boxes=ground_truth
        )
        test_detections = data_util.clip_boxes(
            spec=spec, t_start=train_time, boxes=detections
        )

        new_row: Dict[str, Any] = dict()

        new_row["config_id"] = config_id
        new_row["test_boxes_gt"] = len(test_gt)
        new_row["test_boxes_detected"] = len(test_detections)
        new_row["training_time"] = train_time
        new_row["beta"] = beta

        for threshold_metric in [0.05, 0.1, 0.2, 0.3]:
            recall, recall_dict = metrics.detection_recall(
                ground_truth=test_gt,
                prediction=test_detections,
                threshold=threshold_metric,
                mode="sum",
            )
            precision = metrics.detection_precision(
                ground_truth=test_gt,
                prediction=detections,
                threshold=threshold_metric,
                mode="sum",
            )
            score = metrics.f_beta(precision=precision, recall=recall, beta=beta)
            new_row[f"score_{threshold_metric}"] = score
            new_row[f"recall_{threshold_metric}"] = recall
            new_row[f"precision_{threshold_metric}"] = precision
            new_row[f"recall_dict_{threshold_metric}"] = recall_dict

        new_row.update(config)

        metrics_path = METRICS.joinpath(recording_name)
        df: pd.DataFrame = pd.DataFrame()
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
        df = df.append([new_row])
        df.to_csv(metrics_path, index=False)


def main(
    data_folder: data_util.DataFolder,
    recording_name: str,
    train_time: float,
    beta: float,
    random_search_steps: int,
    num_samples: int,
    save_metrics: bool,
):
    if not NOISES.joinpath(recording_name).exists():
        find_noises_in_recording(data_folder, recording_name)

    for noise in load_noises(recording_name):
        squeak_signal = data_folder.get_signal(name=recording_name)
        print(f"Processing signal `{squeak_signal}`")

        if squeak_signal is None:
            print(f"Signal named {recording_name} not found!!! Skipping...")
            continue

        optimisation_ready = is_configuration_present(
            target_folder=OPTIMISATION_FINAL_RESULTS,
            recording_name=recording_name,
            config_id=noise.config_id,
        )
        annotations_saved = is_configuration_present(
            target_folder=DETECTIONS,
            recording_name=recording_name,
            config_id=noise.config_id,
        )
        if annotations_saved and optimisation_ready:
            print(f"Configuration `{noise.config_id}` already processed. Skipping...")

        spec = prepare_spec(squeak_signal, noise)

        ground_truth = data_util.load_squeak_boxes(
            data_folder, squeak_signal.name, spec
        )

        boxes_noise = data_util.clip_boxes(
            spec=spec, t_start=noise.start, t_end=noise.end, boxes=ground_truth
        )
        if len(boxes_noise) > 0:
            print(f"{len(boxes_noise)} USVs present in the noise area!!! Skipping...")
            continue

        if not optimisation_ready:
            optimise(
                beta,
                ground_truth,
                noise.config_id,
                num_samples,
                random_search_steps,
                recording_name,
                spec,
                train_time,
            )
        else:
            print(f"Skipping GAC optimisation...")

        # detect_and_process_detections(
        #     ground_truth,
        #     noise.config_id,
        #     recording_name,
        #     spec,
        #     train_time,
        #     save_metrics,
        #     beta=beta,
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GAC on a file.")
    parser.add_argument("recording", help="a path to the recording")

    args = parser.parse_args()
    recording = pathlib.Path(args.recording)
    source_folder = recording.parent

    folders = data_util.load_data([source_folder], with_labels=True)
    data_folder: data_util.DataFolder = folders[0]

    for path in [
        NOISES,
        OPTIMISATION_FINAL_RESULTS,
        OPTIMISATION_PARTIAL_RESULTS,
        DETECTIONS,
        METRICS,
    ]:
        if not path.exists():
            path.mkdir(parents=True)

    main(
        data_folder=data_folder,
        recording_name=recording.name,
        beta=BETA,
        train_time=TRAIN_TIME,
        random_search_steps=RANDOM_SEARCH_STEPS,
        num_samples=NUM_SAMPLES,
        save_metrics=SAVE_METRICS,
    )
