import os
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Iterable

import gdown
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN,
    TwoMLPHead,
)
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

from mouse.utils.data_util import merge_boxes, SqueakBox
from mouse.utils.sound_util import SpectrogramData
from .modeling_utils import get_backbone, Normalize

PRETRAINED_MODELS_CHECKPOINTS = {
    "f-rcnn-custom": {
        "url":
            "https://drive.google.com/uc?id=15MM-Qq_KeDLET20aLmrika4qakLTyXOB",  # noqa
        "filename": "test_rcnn.ckpt",
    }
}


def find_USVs(
    spec_data: SpectrogramData,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    confidence_threshold: float = -1,
    silent: bool = False,
    callback: Optional[Callable] = None,
):
    """Load and produce predictions for spectrogram data.

    Parameters
    ----------
    spec_data: SpectrogramData
        Spectrogram data to produce predictions on.
    model_name: str
        Name of pretrained model.
    cache_dir: Path
        Path to where to load or search pretrained models.
    batch_size: int
        Model batch size.
    confidence_threshold: float
        Score threshold to reach to be classified as real detection.
    silent: bool
        Weather to disable tqdm progress bar.
    callback: Optional[Callback]
        Callback called after each iteration.

    Returns
    -------
    List[SqueakBox]
        List of predicted boxes.
    """
    if model_name not in PRETRAINED_MODELS_CHECKPOINTS:
        raise ValueError(f"{model_name} is not a pretrained model name")

    model_path = cache_dir.joinpath(
        PRETRAINED_MODELS_CHECKPOINTS[model_name]["filename"])
    if not model_path.exists():
        os.makedirs(str(cache_dir), exist_ok=True)
        gdown.download(PRETRAINED_MODELS_CHECKPOINTS[model_name]["url"],
                       output=str(model_path))

    model = USVDetector.load_from_checkpoint(str(model_path), inference_only=True)
    return model.predict_for_spectrogram_data(
        spec_data,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        silent=silent,
        callback=callback,
    )


def _preprocess_spec(spec, freqs, use_log, clip_18khz, gamma):
    if use_log:
        spec[spec == 0] = 1e-8
    if use_log is True or use_log == 10:
        spec = np.log10(spec)
    elif use_log == 2:
        spec = np.log2(spec)
    elif use_log == "e":
        spec = np.log(spec)
    elif use_log is not False:
        # calculate log with arbitrary base
        spec = np.log(spec) / np.log(use_log)

    freq_offset = 0
    if clip_18khz:
        freq_mask = freqs >= 18000
        freq_offset = freqs.shape[0] - np.sum(freq_mask)
        freqs = freqs[freq_mask]
        spec = spec[freq_mask, :]

    return spec**gamma, freqs, freq_offset


# pruned version of model, full train loop is published in other repo
class USVDetector(pl.LightningModule):
    """General purpose model used to load serialized model and produce predictions.

    Parameters
    ----------
    args: Namespace
        Used to reproduce model architecture. (loaded from checkpoint)
    inference_only: bool
        When True loads model in inference environment
        with no access to pretrained backbone
    """

    def __init__(self, args: Namespace, inference_only: bool = False):
        super().__init__()
        self.save_hyperparameters(args)
        backbone = get_backbone(
            model_name=self.hparams.backbone,
            path=self.hparams.backbone_pretrained_path,
            inference_only=inference_only,
        )

        self.model = self._construct_model(backbone)

    def _construct_model(self, backbone):
        if self.hparams.model_type == "rcnn":
            anchor_generator = AnchorGenerator(
                sizes=(self.hparams.sizes,),
                aspect_ratios=(self.hparams.aspect_rations,),
            )
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0"],
                output_size=self.hparams.roi_output_size,
                sampling_ratio=self.hparams.sampling_ratio,
            )

            box_predictor = FastRCNNPredictor(self.hparams.representation_size,
                                              self.hparams.num_classes)

            resolution = roi_pooler.output_size[0]
            box_head = TwoMLPHead(
                backbone.out_channels * resolution**2,
                self.hparams.representation_size,
            )

            model = FasterRCNN(
                backbone,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                box_head=box_head,
                box_predictor=box_predictor,
                image_mean=(0,),
                image_std=(1,),
                max_size=1000,
                min_size=220,
            )
            model.transform = Normalize(self.hparams.data_mean, self.hparams.data_std)
        else:
            raise ValueError(f"Unknown model type {self.hparams.model_type}")

        return model

    def forward(self, specs: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """Modules forward method."""
        return self.model(specs)

    def combine_and_filter_predictions(
        self,
        all_preds: Iterable[Dict[str, Tensor]],
        stride: int,
        confidence_threshold: float,
    ):
        """Combine batched model predictions into one (undo batching).

        Parameters
        ----------
        all_preds: Iterable[Dict[str, Tensor]]
            List of batched model predictions.
        stride: int
            Stride of sliding window used to generate spectrogram chunks.
        confidence_threshold: float
            Score threshold to reach to be classified as real detection.

        Returns
        -------
        np.ndarray
            All predicted boxes with corrected time.
        np.ndarray
            All labels predicted by model (corresponds to each box).
        np.ndarray
            All confidence scores produced by model
            (corresponds to returned boxes).
        """
        all_boxes = []
        all_labels = []
        all_scores = []

        offset = 0
        for chunk_id, pred in enumerate(all_preds):
            boxes, labels = (
                pred["boxes"].detach().cpu().numpy().astype(np.int64),
                pred["labels"].detach().cpu().numpy(),
            )

            if "scores" in pred:
                scores = pred["scores"].detach().cpu().numpy()
                boxes = boxes[scores > confidence_threshold]
                labels = labels[scores > confidence_threshold]
                scores = scores[scores > confidence_threshold]
                all_scores.append(scores)

            if len(boxes) > 0:
                boxes[:, 0] += offset
                boxes[:, 2] += offset
            offset += stride

            all_boxes.append(boxes)
            all_labels.append(labels)

        if all_boxes:
            all_boxes = np.concatenate(all_boxes)
        if all_labels:
            all_labels = np.concatenate(all_labels)
        if all_scores:
            all_scores = np.concatenate(all_scores)

        return all_boxes, all_labels, all_scores

    def _merge_boxes_for_label(
        self,
        spec_data: SpectrogramData,
        squeaks: List[SqueakBox],
        delta_freq=0.0,
        delta_time=0.01,
        label_id=None,
        label=None,
    ):
        """Merge boxes for specific label."""
        return merge_boxes(
            spec_data,
            [squeak for squeak in squeaks if squeak.label == label_id],
            delta_freq=delta_freq,
            delta_time=delta_time,
            label=label,
        )

    def _merge_spec_predictions(
        self,
        spec_data: SpectrogramData,
        all_preds: List[Dict[str, Tensor]],
        stride: int,
        confidence_threshold: float,
        freq_offset: int,
    ):
        all_boxes, all_labels, _ = self.combine_and_filter_predictions(
            all_preds, stride, confidence_threshold
        )

        squeaks = [
            SqueakBox(
                freq_start=freq_start,
                freq_end=freq_end,
                t_start=t_start,
                t_end=t_end,
                label=label,
            ) for (t_start, freq_start, t_end, freq_end),
            label in zip(  # noqa
                all_boxes, all_labels)
        ]

        # only merge predictions for same label
        predicted_boxes = []
        for label_id, label in enumerate(self.hparams.labels[1:], 1):
            predicted_label_boxes = self._merge_boxes_for_label(
                spec_data=spec_data,
                squeaks=squeaks,
                delta_freq=0.0,
                delta_time=0.01,
                label_id=label_id,
                label=label,
            )

            predicted_label_boxes = [
                SqueakBox(
                    freq_start=box.freq_start + freq_offset,
                    freq_end=box.freq_end + freq_offset,
                    t_start=box.t_start,
                    t_end=box.t_end,
                    label=box.label,
                ) for box in predicted_label_boxes
            ]

            predicted_boxes.extend(predicted_label_boxes)
        return predicted_boxes

    def predict_for_spectrogram_data(
        self,
        spec_data: SpectrogramData,
        batch_size: Optional[int] = None,
        confidence_threshold: float = -1,
        silent: bool = False,
        callback: Optional[Callable] = None,
    ) -> List[SqueakBox]:
        """Use to produce and process model predictions.

        Parameters
        ----------
        spec_data: SpectrogramData
            Spectrogram data to produce predictions on.
        batch_size: int
            Model batch size.
        confidence_threshold: float
            Score threshold to reach to be classified as real detection.
        silent: bool
            Weather to disable tqdm progress bar.
        callback: Optional[Callback]
            Callback called after each iteration.

        Returns
        -------
        List[SqueakBox]
            List of predicted boxes.
        """
        spec = spec_data.spec
        times = spec_data.times
        freqs = spec_data.freqs

        spec, freqs, freq_offset = _preprocess_spec(
            spec,
            freqs,
            self.hparams.use_log,
            not self.hparams.no_clip_18khz,
            self.hparams.gamma,
        )

        example_duration = self.hparams.example_duration
        overlap = self.hparams.overlap

        if isinstance(example_duration, float):
            time_delta = times[1] - times[0]
            example_duration = int(np.ceil(example_duration / time_delta))

        if isinstance(overlap, float):
            stride = int((1 - overlap) * example_duration)
        else:
            stride = example_duration - overlap

        chunks = [
            torch.tensor(spec[:,
                              chunk_start:chunk_start + example_duration]).unsqueeze(0)
            for chunk_start in range(0, times.shape[0] - example_duration, stride)
        ]

        batch_size = batch_size if batch_size else self.hparams.eval_batch_size
        dataloader = DataLoader(chunks, batch_size=batch_size)
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, disable=silent)):
                preds = self.model(batch)
                all_preds.extend(preds)
                if callback:
                    callback(idx, len(dataloader))

        pred_box = self._merge_spec_predictions(
            SpectrogramData(spec, times, freqs),
            all_preds,
            stride,
            self.hparams.confidence_threshold
            if confidence_threshold == -1 else confidence_threshold,
            freq_offset,
        )
        return pred_box
