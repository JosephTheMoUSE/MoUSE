import os
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import time
import argparse
import pathlib
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import uuid
import warnings
import gc
import functools

import numpy as np
from skimage import segmentation

import mouse
import mouse.segmentation
from mouse.denoising import denoising
from mouse.segmentation import balloon_from_latent, threshold_from_latent
from mouse.utils import data_util, sound_util, metrics
from mouse.utils.data_util import SignalNoise
from mouse.utils.metrics import Metric
from mouse.utils import metrics
from mouse.utils.data_util import SqueakBox
from mouse.utils.sound_util import SpectrogramData

NOISES = pathlib.Path("noises")
RESULTS_ROOT = pathlib.Path("optimisation_results")
OPTIMISATION_FINAL_RESULTS = RESULTS_ROOT.joinpath("final")
OPTIMISATION_PARTIAL_RESULTS = RESULTS_ROOT.joinpath("partial")
DETECTIONS = RESULTS_ROOT.joinpath("detections")
METRICS = RESULTS_ROOT.joinpath("metrics")

# address = os.environ.get('RAY_HEAD_ADDRESS')
# redis_pass = os.environ.get('RAY_REDIS_PASS')

# ray.init(address=address, redis_pass=redis_pass)
ray.init(address='ray://172.29.96.11:10001')

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

GAC_SEARCH_SPACE = {
    "iterations": tune.uniform(1, 40),
    "smoothing": tune.uniform(0.0, 7.99),
    "flood_threshold": tune.uniform(0.1, 0.99),
    "_balloon_latent": tune.uniform(0.2, 1.8),
    "sigma": tune.uniform(1, 10),
}


def _test_gac_config(
        config: Dict[str, float],
        ground_truth: List[SqueakBox],
        spec: SpectrogramData,
        alpha: float,
        threshold_metric: float,
        metric: Metric,
        beta: Optional[float] = None,
):
    level_set = mouse.segmentation.eroded_level_set_generator(config["flood_threshold"])

    boxes = mouse.segmentation.find_USVs(
        spec=spec,
        min_side_length=1,
        filter=True,
        iterations=int(config["iterations"]),
        smoothing=int(config["smoothing"]),
        threshold=threshold_from_latent(config["_balloon_latent"]),
        balloon=balloon_from_latent(config["_balloon_latent"]),
        level_set=level_set,
        preprocessing_fn=partial(segmentation.inverse_gaussian_gradient,
                                 sigma=config["sigma"],
                                 alpha=alpha),
    )

    score = 0.0
    precision: Optional[float] = None
    recall: Optional[float] = None
    recall_dict: Dict[str, float] = dict()
    if len(boxes) != 0:
        if metric == Metric.F_BETA:
            if beta is None:
                raise ValueError(f"When `{Metric.F_BETA}` metric is used, "
                                 f"`beta` can't be None!")
            recall, recall_dict = metrics.detection_recall(
                ground_truth=ground_truth,
                prediction=boxes,
                threshold=threshold_metric,
                mode="sum",
            )
            precision = metrics.detection_precision(
                ground_truth=ground_truth,
                prediction=boxes,
                threshold=threshold_metric,
                mode="sum",
            )
            score = metrics.f_beta(precision=precision, recall=recall, beta=beta)
        elif metric == Metric.IOU:
            score = metrics.intersection_over_union_global(ground_truth=ground_truth,
                                                           prediction=boxes)
        else:
            raise ValueError(f"`metric` should be one of "
                             f"{[v for v in Metric]} but found {metric}")

    tune.report(
        score=score,
        precision=precision,
        recall=recall,
        recall_dict=recall_dict,
        box_count=len(boxes),
        boxes=repr(boxes),
    )


@ray.remote(num_cpus=1)
def optimize_file(spec_data, boxes):
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=1)
    # trainable_with_resources = tune.with_resources(objective, {"cpu": 2})
    res = tune.run(
        tune.with_parameters(
            tune.with_resources(_test_gac_config, {"cpu": 1}),
            spec=spec_data,
            ground_truth=boxes,
            beta=BETA,
            alpha=ALPHA,
            metric=Metric.F_BETA,
            threshold_metric=0.1,
        ),
        metric='score',
        mode='max',
        search_alg=algo,
        num_samples=200,
        config=GAC_SEARCH_SPACE
    )
    return res.get_best_result().config


def find_noises_in_recording(data_folder, signal_data,
                             min_noise_sample_duration: float = 1.0):
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
                    weights = np.array(
                        [np.mean(np.quantile(spec_data.spec[:, s:e], 0.75, axis=1)) for s, e in spans if e - s > 1])
                    span = spans[np.random.choice(len(spans), p=weights / np.sum(weights))]
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
    return [
        SignalNoise(config_id=config_id,
                    start=time_info["start"],
                    end=time_info["end"],
                    noise_decrease=time_info["noise_decrease"])
        for config_id, time_info in noise_config_results.items()
    ]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test GAC on a file.")
    parser.add_argument("dataset_dir", help="a path to the recording")
    args = parser.parse_args()

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

    futures = []
    for data_folder in folders:
        recording_name = data_folder.wavs[0].name
        signal_data = data_folder.get_signal(name=recording_name)
        for noise in find_noises_in_recording(data_folder, signal_data):
            spec = prepare_spec(signal_data, noise)

            ground_truth = data_util.load_squeak_boxes(
                data_folder, signal_data.name, spec
            )

            boxes_noise = data_util.clip_boxes(
                spec=spec, t_start=noise.start, t_end=noise.end, boxes=ground_truth
            )

            spec_id = ray.put(spec)
            futures.append(optimize_file.remote(spec_id, boxes_noise))
            gc.collect()

    print(ray.get(futures))