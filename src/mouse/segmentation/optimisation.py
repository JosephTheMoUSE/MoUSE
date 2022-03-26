"""Module for automatic GAC configuration."""
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from skimage import segmentation

import mouse
from mouse.utils import metrics
from mouse.utils.data_util import SqueakBox
from mouse.utils.metrics import Metric
from mouse.utils.sound_util import SpectrogramData

GAC_SEARCH_SPACE = {
    "iterations": tune.uniform(1, 40),
    "smoothing": tune.uniform(0.0, 7.99),
    "flood_threshold": tune.uniform(0.1, 0.99),
    "threshold": tune.uniform(0.2, 0.99),
    "sigma": tune.uniform(1, 10),
}


def _test_gac_config(
    config: Dict[str, float],
    ground_truth: List[SqueakBox],
    spec: SpectrogramData,
    balloon: int,
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
        threshold=config["threshold"],
        balloon=balloon,
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


def optimise_gac(
    spec: SpectrogramData,
    ground_truth: List[SqueakBox],
    alpha: float,
    balloon: float,
    num_samples: int,
    random_search_steps: int,
    max_concurrent: int,
    metric: Metric,
    threshold_metric: float,
    beta: Optional[float],
    callbacks: Optional[Sequence[tune.Callback]],
) -> tune.ExperimentAnalysis:
    """Search for best GAC parameters with tune.

    Parameters
    ----------
    spec: SpectrogramData
        Spectrogram on which GAC will be optimised.

    ground_truth: List[SqueakBox]
        USVs to optimise for.

    alpha: float
        `alpha` parameter of GAC. This parameter is not optimised.

    balloon: float
        `balloon` parameter of GAC. This parameter is not optimised.

    num_samples: int
        Number of tests performed during optimisation.

    random_search_steps: int
        Number of random tests. Random tests initialize optimisation process.

    max_concurrent: int
        Number of concurrent tests.

    metric: Metric
        Metric maximised during optimisation. Allowed values are defined by `Metric`
        enum.

    threshold_metric: float
        Threshold for area-based determining which squeaks were detected. Used for
        metric calculation (precision, recall).

    beta: Optional[float]
        Parameter of 'f_beta' metric.

    callbacks: Optional[Sequence[tune.Callback]]
        Callbacks called after every test.

    Returns
    -------
    tune.ExperimentAnalysis
        Optimisation result.
    """
    analysis = tune.run(
        tune.with_parameters(
            _test_gac_config,
            spec=spec,
            ground_truth=ground_truth,
            balloon=balloon,
            beta=beta,
            alpha=alpha,
            metric=metric,
            threshold_metric=threshold_metric,
        ),
        config=GAC_SEARCH_SPACE,
        verbose=0,
        metric="score",
        mode="max",
        callbacks=callbacks,
        search_alg=ConcurrencyLimiter(
            BayesOptSearch(random_search_steps=random_search_steps),
            max_concurrent=max_concurrent,
        ),
        num_samples=num_samples,
        local_dir=tempfile.gettempdir(),
        name="mouse/GAC_optimisation",
    )
    shutil.rmtree(Path(tempfile.gettempdir()).joinpath("mouse"))

    return analysis
