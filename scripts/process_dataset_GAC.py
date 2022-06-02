from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
from functools import partial
from typing import Any
from typing import Dict, List

import pandas as pd
import numpy as np
import warnings
import uuid
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
from multiprocessing import Pool

# Constants
# # Paths
NOISES = pathlib.Path("noises")
RESULTS_ROOT = pathlib.Path("optimisation_results")
OPTIMISATION_FINAL_RESULTS = RESULTS_ROOT.joinpath("final")
OPTIMISATION_PARTIAL_RESULTS = RESULTS_ROOT.joinpath("partial")
DETECTIONS = RESULTS_ROOT.joinpath("detections")
METRICS = RESULTS_ROOT.joinpath("metrics")

# # Optimisation-related constants
MAX_CONCURRENT = 1  # max number of concurrent optimisation trials
TRAIN_TIME = 10.0
TIME_DELTA = 0.01
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

# # Noise search
NUM_NOISE_CONFIGURATIONS = 3


def load_noises(recording_name: str) -> List[SignalNoise]:
    with NOISES.joinpath(recording_name).open("r") as fp:
        noise_times: Dict[str, dict] = json.load(fp)

    return [
        SignalNoise(config_id=config_id,
                    start=time_info["start"],
                    end=time_info["end"],
                    noise_decrease=time_info["noise_decrease"])
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


def find_noises_in_recording(data_folder: data_util.DataFolder, recording_name: str,
                             min_noise_sample_duration: float = 1.0):
    """Create .json noise file listing noises from recording.

    Create noise file for recording named `recording_name`. The file is created in
    `NOISES` folder and it's named `recording_name`.

    Format of this file is as follows:
    {
        <configuration_identifier_1>: {"start": <noise start [seconds]>,
                                       "end": <noise end [seconds]>,
                                       "noise_decrease": <noise_decrease [%]>},
        <configuration_identifier_2>: {"start": <noise start [seconds]>,
                                       "end": <noise end [seconds]>,
                                       "noise_decrease": <noise_decrease [%]>},
        ...
    }
    """
    signal_data = data_folder.get_signal(name=recording_name)
    spec_data = sound_util.spectrogram(
        signal=signal_data.signal,
        sample_rate=signal_data.sample_rate
    )
    boxes = data_util.load_squeak_boxes(
        data_folder, signal_data.name, spec_data
    )
    min_noise_sample_len = spec_data.time_to_pixels(min_noise_sample_duration)

    noise_config_results = {}
    for i in range(NUM_NOISE_CONFIGURATIONS):
        noise_decrease = 0.5

        if boxes:
            spans = []
            boxes = sorted(boxes, key=lambda b: b.t_start)
            current_end = 0
            for box in boxes:
                d = current_end - box.t_start
                if d >= min_noise_sample_len:
                    spans.append([current_end, box.t_start])
                    current_end = box.t_end
                elif d >= 0:
                    current_end = box.t_end
                elif current_end < box.t_end:
                    current_end = box.t_end
            if spans:
                try:
                    weights = np.array([np.mean(np.quantile(spec_data.spec[:, s:e], 0.75, axis=1)) for s, e in spans if e - s > 1])
                    span = spans[np.random.choice(len(spans), p=weights/np.sum(weights))]
                except:
                    span = [0, spec_data.times.shape[0] - 1]
                    noise_decrease = noise_decrease / 2
                    warnings.warn(f"Error occurred while searching for noise. For file {recording_name}\n"
                                  "Span will be chosen randomly. Decreasing `noise_decrease` by half")
            else:
                # TODO: fall back strategy
                span = [0, spec_data.times.shape[0] - 1]
                noise_decrease = noise_decrease / 2
                warnings.warn(f"Can't find sufficiently long span without calls. For file {recording_name}\n"
                              "Span will be chosen randomly. Decreasing `noise_decrease` by half")
        else:
            span = [0, spec_data.times.shape[0] - 1]

        if span[1] - span[0] > min_noise_sample_len:
            span[0] = np.random.randint(span[0], span[1] - min_noise_sample_len + 1)
            span[1] = span[0] + min_noise_sample_len

        noise_config_results[str(uuid.uuid4())] = {
            "start": spec_data.times[span[0]],
            "end": spec_data.times[span[1]],
            "noise_decrease": noise_decrease
        }

    with NOISES.joinpath(recording_name).open("w") as f:
        json.dump(noise_config_results, f)


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
        noise_decrease=noise.noise_decrease,
    )
    spec.spec = spec.spec.float()
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


def extract_train_data(spec,
                       max_time,
                       boxes):

    if len(boxes) == 0:
        return *data_util.clip_spec_and_boxes(spec, boxes, t_end=max_time/2), max_time/2

    si_boxes = [data_util.SqueakBoxSI.from_squeak_box(box, spec) for box in boxes]
    events = sorted([(box.t_start, False, box) for box in si_boxes] + [(box.t_end, True, box) for box in si_boxes])

    box_endings = []
    box_endings_idx = 0

    max_boxes_in_window = 0
    opt_start = None

    curr_boxes = 0
    curr_start = None
    potential_starts = []
    starts_idx = 0

    for pos, is_end, box in events:
        if is_end:
            box_endings.append(pos)
            continue
        else:
            if curr_start is None:
                curr_start = box
                curr_boxes += 1
                while box_endings_idx < len(box_endings) and curr_start.t_start > box_endings[box_endings_idx]:
                    curr_boxes -= 1
                    box_endings_idx += 1
            else:
                potential_starts.append(box)

        if pos - curr_start.t_start > max_time:
            if starts_idx == len(potential_starts):
                curr_start = None
                continue
            curr_start = potential_starts[starts_idx]
            starts_idx += 1
            while box_endings_idx < len(box_endings) and curr_start.t_start > box_endings[box_endings_idx]:
                curr_boxes -= 1
                box_endings_idx += 1
        else:
            curr_boxes += 1

        if curr_boxes > max_boxes_in_window:
            max_boxes_in_window = curr_boxes
            opt_start = curr_start

    t_start = max(opt_start.t_start - TIME_DELTA, 0)
    t_end = t_start + max_time
    print(f"Found optimal train window at {t_start}:{t_end}s")
    return *data_util.clip_spec_and_boxes(spec, boxes, t_start=t_start, t_end=t_end), max_time


def optimise(
        beta,
        ground_truth,
        config_id,
        num_samples,
        random_search_steps,
        recording_name,
        spec,
        max_train_time,
):
    spec_train, ground_truth_train, train_time = extract_train_data(
        spec=spec, max_time=max_train_time, boxes=ground_truth
    )
    print(f"Number of training USVs: {len(ground_truth_train)}")
    if len(ground_truth_train) == 0:
        warnings.warn("No USVs in training area!!!")

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
        detect: bool
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
            warnings.warn(f"{len(boxes_noise)} USVs present in the noise area!!! "
                          f"Change script to overwrite default behavior")
            # continue

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

        if detect:
            detect_and_process_detections(
                ground_truth,
                noise.config_id,
                recording_name,
                spec,
                train_time,
                save_metrics,
                beta=beta,
            )


def process_recording(recording, data_folder, args):
    main(
        data_folder=data_folder,
        recording_name=recording.name,
        beta=BETA,
        train_time=TRAIN_TIME,
        random_search_steps=RANDOM_SEARCH_STEPS,
        num_samples=NUM_SAMPLES,
        save_metrics=(not args.no_save_metrics),
        detect=args.detect
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GAC on a file.")
    parser.add_argument("dataset_dir", help="a path to the recording")
    parser.add_argument('--no_save_metrics', action='store_true')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--worker_count', default=os.cpu_count(), type=int)
    parser.add_argument('--chunk_num', default=0, type=int)
    parser.add_argument('--chunk_count', default=3, type=int)
    args = parser.parse_args()

    #worker_count = args.worker_count
    MAX_CONCURRENT = args.worker_count

    source_folder = pathlib.Path(args.dataset_dir)
    source_folders = [d for d in source_folder.iterdir()
                      if d.is_dir() and not d.name.startswith('.')
                      and NOISES.name not in d.name
                      and RESULTS_ROOT.name not in d.name
                      and OPTIMISATION_FINAL_RESULTS.name not in d.name
                      and OPTIMISATION_PARTIAL_RESULTS.name not in d.name
                      and DETECTIONS.name not in d.name
                      and METRICS.name not in d.name]

    folders = sorted(data_util.load_data(source_folders, with_labels=True), key=lambda x: x.folder_path)
    l = len(folders)
    chunk = l // args.chunk_count
    folders = folders[chunk*args.chunk_num:chunk*(args.chunk_num+1)]
    #data_folder: data_util.DataFolder = folders[0]

    NOISES = source_folder.joinpath(NOISES)
    OPTIMISATION_FINAL_RESULTS = source_folder.joinpath(OPTIMISATION_FINAL_RESULTS)
    OPTIMISATION_PARTIAL_RESULTS = source_folder.joinpath(OPTIMISATION_PARTIAL_RESULTS)
    DETECTIONS = source_folder.joinpath(DETECTIONS)
    METRICS = source_folder.joinpath(METRICS)

    for path in [
        NOISES,
        OPTIMISATION_FINAL_RESULTS,
        OPTIMISATION_PARTIAL_RESULTS,
        DETECTIONS,
        METRICS,
    ]:
        if not path.exists():
            path.mkdir(parents=True)

    #process = functools.partial(process_recording, data_folder=data_folder, args=args)
    for data_folder in folders:
        recording = data_folder.wavs[0]
        try:
            process_recording(recording, data_folder=data_folder, args=args)
        except:
            warnings.warn(f'Encountered error with file {recording}')
    #with Pool(args.worker_count) as p:
    #    p.map(process, recordings)
