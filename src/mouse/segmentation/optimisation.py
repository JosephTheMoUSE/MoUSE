"""Module for automatic GAC configuration."""
import shutil
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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
    "iterations": (1, 40),
    "smoothing": (0.0, 7.99),
    "flood_threshold": (0.1, 0.99),
    "_balloon_latent": (0.2, 1.8),
    "sigma": (1, 10),
}
GAC_SEARCH_SPACE_TUNE = {k: tune.sample.Float(*v) for k, v in GAC_SEARCH_SPACE.items()}


def balloon_from_latent(balloon_latent: float) -> int:
    """Extract balloon parameter from latent variable."""
    if balloon_latent > 1:
        return -1
    return 1


def threshold_from_latent(balloon_latent: float) -> float:
    """Extract threshold parameter from latent variable."""
    if balloon_latent > 1:
        return 2 - balloon_latent
    return balloon_latent


def _test_gac_config(
    ground_truth: List[SqueakBox],
    spec: SpectrogramData,
    alpha: float,
    threshold_metric: float,
    metric: Metric,
    beta: Optional[float] = None,
    **kwargs,
):
    use_tune = True
    if 'config' not in kwargs:
        use_tune = False

    if use_tune:
        config = kwargs['config']
    else:
        config = kwargs

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
    if not use_tune:
        return score
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
    num_samples: int,
    random_search_steps: int,
    max_concurrent: int,
    metric: Metric,
    threshold_metric: float,
    beta: Optional[float],
    callbacks: Optional[Sequence[tune.Callback]],
    use_ray: bool = True,
    remove_ray_results: bool = True,
) -> Union[tune.ExperimentAnalysis, None]:
    """Search for best GAC parameters with tune.

    Parameters
    ----------
    spec: SpectrogramData
        Spectrogram on which GAC will be optimised.

    ground_truth: List[SqueakBox]
        USVs to optimise for.

    alpha: float
        `alpha` parameter of GAC. This parameter is not optimised.

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

    use_ray: bool
        Specifies whether optimisation should use ray. This needs to be False only when
        the app is packaged with PyInstaller.

    remove_ray_results: bool
        Specifies whether optimisation results will be deleted from disk. Should be
        `False` if many optimisation instances are executed.

    Returns
    -------
    Union[tune.ExperimentAnalysis, None]
        Optimisation result is returned when `use_ray` is True.
    """
    _test_gac_config_with_parameters = partial(
        _test_gac_config,
        spec=spec,
        ground_truth=ground_truth,
        beta=beta,
        alpha=alpha,
        metric=metric,
        threshold_metric=threshold_metric,
    )
    if use_ray:
        analysis = tune.run(
            tune.with_parameters(
                _test_gac_config,
                spec=spec,
                ground_truth=ground_truth,
                beta=beta,
                alpha=alpha,
                metric=metric,
                threshold_metric=threshold_metric,
            ),
            config=GAC_SEARCH_SPACE_TUNE,
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
    else:
        import bayes_opt
        optimizer = bayes_opt.BayesianOptimization(
            f=_test_gac_config_with_parameters,
            pbounds=GAC_SEARCH_SPACE,
        )

        def _optimisation_callback(event, instance: bayes_opt.BayesianOptimization):
            result: Dict[str, Union[dict, float]] = defaultdict(lambda: dict())

            result["score"] = instance.space.target[-1]
            result["precision"] = 0
            result["recall"] = 0
            result["box_count"] = 0
            for name, value in zip(instance.space.keys, instance.space.params[-1]):
                result["config"][name] = value

            for callback in callbacks:
                callback.on_trial_result(iteration=None,
                                         trials=None,
                                         trial=None,
                                         result=result)

        optimizer.subscribe(
            event=bayes_opt.Events.OPTIMIZATION_STEP,
            subscriber="_optimisation_callback",
            callback=_optimisation_callback,
        )
        optimizer.maximize(
            init_points=random_search_steps,
            n_iter=num_samples - random_search_steps,
        )

    if use_ray and remove_ray_results:
        shutil.rmtree(Path(tempfile.gettempdir()).joinpath("mouse"))

    if use_ray:
        return analysis
    return None
