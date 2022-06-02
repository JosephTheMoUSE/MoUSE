"""Module implementing CNN-based USVs classification."""
from argparse import Namespace
from typing import Tuple, List, Optional, Callable
from copy import deepcopy as copy

import os
import gdown
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm

from mouse.utils.data_util import SqueakBox
from mouse.utils.sound_util import SpectrogramData
from mouse.utils.modeling_utils import get_backbone

PRETRAINED_MODELS_CHECKPOINTS = {
    "cnn-binary-v1-custom": {
        "url":
            "https://drive.google.com/uc?id=1XhAyU6aFJoWZn8pxI-N9BRnpUTGMjp4Z",  # noqa
        "filename": "cnn_binary_v1_custom.ckpt",
    }
}


def classify_USVs(
    spec_data: SpectrogramData,
    annotations: List[SqueakBox],
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
    annotations: List[SqueakBox]
        List of annotations to run classification on.
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
        List of annotations after classification.
    """
    if model_name not in PRETRAINED_MODELS_CHECKPOINTS:
        raise ValueError(f"{model_name} is not a pretrained model name")

    model_path = cache_dir.joinpath(
        PRETRAINED_MODELS_CHECKPOINTS[model_name]["filename"])
    if not model_path.exists():
        os.makedirs(str(cache_dir), exist_ok=True)
        gdown.download(PRETRAINED_MODELS_CHECKPOINTS[model_name]["url"],
                       output=str(model_path))

    model = USVClassifier.load_from_checkpoint(str(model_path), inference_only=True)
    return model.predict_for_annotations(
        spec_data,
        annotations=annotations,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        silent=silent,
        callback=callback,
    )


class USVClassifier(pl.LightningModule):
    """General purpose CNN classification model used to load serialized and classify USVs.

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

        self.backbone = get_backbone(model_name=self.hparams.backbone,
                                     path=None,
                                     inference_only=inference_only)

        if self.hparams.add_pos_embeddings:
            self.cls = nn.Linear(in_features=self.backbone.out_channels + 4,
                                 out_features=self.hparams.num_classes)
        else:
            self.cls = nn.Linear(in_features=self.backbone.out_channels,
                                 out_features=self.hparams.num_classes)

    def forward(self, specs, boxes=None) -> Tensor:
        """Modules forward propagation method."""
        feats = self.backbone(specs)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.view(feats.shape[0], -1)
        if boxes and self.hparams.add_pos_embeddings:
            pos = [(min(box.freq_start, self.hparams.max_pos) * np.pi / 2 /
                    self.hparams.max_pos,
                    min(box.freq_end, self.hparams.max_pos) * np.pi / 2 /
                    self.hparams.max_pos) for box in boxes]
            pos_embeds = torch.tensor(
                [(np.cos(p[0]), np.sin(p[0]), np.cos(p[1]), np.sin(p[1])) for p in pos],
                device=feats.device,
                dtype=torch.float32)
            return F.softmax(self.cls(torch.hstack((feats, pos_embeds))), dim=1)
        else:
            return F.softmax(self.cls(feats), dim=1)

    def predict_for_annotations(self,
                                spec_data: SpectrogramData,
                                annotations: List[SqueakBox],
                                batch_size: Optional[int] = None,
                                confidence_threshold: float = -1,
                                silent: bool = False,
                                callback: Optional[Callable] = None) -> List[SqueakBox]:
        """Use to classify annotations.

        Parameters
        ----------
        spec_data: SpectrogramData
            Spectrogram data to produce predictions on.
        annotations: List[SqueakBox]
            List of annotations to run classification on.
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
            List of annotations after classification.
        """
        max_call_len = self.hparams.max_call_len
        stride = self.hparams.stride
        hop_len = int(max_call_len * stride)
        add_time_pixel_context = self.hparams.add_time_pixel_context
        add_freq_pixel_context = self.hparams.add_freq_pixel_context

        examples: List[SqueakBox] = []
        preds: List[Tuple[int, int, float]] = []
        example_map: List[int] = []

        for idx, annotation in enumerate(tqdm(annotations, disable=silent)):
            if annotation.t_end - annotation.t_start <= max_call_len:
                annotation.t_start = max(0, annotation.t_start - add_time_pixel_context)
                annotation.t_end = min(spec_data.times.shape[-1],
                                       annotation.t_end + add_time_pixel_context)
                annotation.freq_start = max(
                    0, annotation.freq_start - add_freq_pixel_context)
                annotation.freq_end = min(spec_data.freqs.shape[-1],
                                          annotation.freq_end + add_freq_pixel_context)
                examples.append(annotation)
                example_map.append(idx)
            else:
                for start in range(annotation.t_start,
                                   annotation.t_end - max_call_len + 1,
                                   int(hop_len)):
                    chunk = copy(annotation)
                    chunk.t_start = start
                    chunk.t_end = start + max_call_len
                    chunk.t_start = max(0, chunk.t_start - add_time_pixel_context)
                    chunk.t_end = min(spec_data.times.shape[-1],
                                      chunk.t_end + add_time_pixel_context)
                    chunk.freq_start = max(0, chunk.freq_start - add_freq_pixel_context)
                    chunk.freq_end = min(spec_data.freqs.shape[-1],
                                         chunk.freq_end + add_freq_pixel_context)
                    examples.append(chunk)
                    example_map.append(idx)

            if callback:
                callback(idx, len(annotations))

            if len(examples) >= batch_size:
                examples_to_process = examples[:batch_size]
                examples_to_process_map = example_map[:batch_size]
                examples = examples[batch_size:]
                example_map = example_map[batch_size:]

                images = self._construct_model_input(spec_data, examples_to_process)
                with torch.no_grad():
                    scores, pred = torch.max(self(images, examples_to_process).detach(), dim=1)
                    scores = scores.cpu().numpy()
                    pred = pred.cpu().numpy()
                preds.extend([(idx, p, s) for idx,
                              p,
                              s in zip(examples_to_process_map, pred, scores)])
        while examples:
            examples_to_process = examples[:batch_size]
            examples_to_process_map = example_map[:batch_size]
            examples = examples[batch_size:]
            example_map = example_map[batch_size:]

            images = self._construct_model_input(spec_data, examples_to_process)
            with torch.no_grad():
                scores, pred = torch.max(self(images, examples_to_process).detach(), dim=1)
                scores = scores.cpu().numpy()
                pred = pred.cpu().numpy()
            preds.extend([(idx, p, s)
                          for idx,
                          p,
                          s in zip(examples_to_process_map, pred, scores)
                          if s >= confidence_threshold])
        max_score = 0
        pred = 0
        for i, (idx, p, s) in enumerate(preds + [(-1, -1, -1)]):
            if idx == -1:
                break
            if s > max_score:
                pred = p
            if i + 1 == len(preds) or idx != preds[i + 1][0]:
                annotations[idx].label = self.hparams.labels[pred]
                max_score = 0
                pred = 0
        return annotations

    def _construct_model_input(self,
                               spec_data: SpectrogramData,
                               examples_to_process: List[SqueakBox]):
        box_contents = [
            spec_data.spec[box.freq_start:box.freq_end, box.t_start:box.t_end]
            for box in examples_to_process
        ]
        H = max([box_content.shape[0] for box_content in box_contents] + [32])
        W = max([box_content.shape[1] for box_content in box_contents] + [32])
        images = torch.zeros(len(box_contents), 1, H, W, dtype=torch.float32)
        for idx, box_content in enumerate(box_contents):
            images[idx, 0, 0:box_content.shape[0], 0:box_content.shape[1]] = \
                (box_content - self.hparams.data_mean[0]) / self.hparams.data_std[0]
        return images.to(self.device)
