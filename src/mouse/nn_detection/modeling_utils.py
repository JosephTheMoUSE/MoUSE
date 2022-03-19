"""Module implementing NN utilities."""
import math
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.models.detection.image_list import ImageList


class BackboneBlock(nn.Module):
    """VGG-like model block used to construct custom model.

    Parameters
    ----------
    in_channels : int
        Number of channels that come to this layer.
    out_channels : int
        Number of convolutional filters to use in this layer.
    kernel_size : Tuple[int]
        Size of convolutional filters.
    stride : Union[int, Tuple]
        Convolution stride.
    padding : Union[int, Tuple, str]
        Specifies padding size (if applicable).
        Refer to torch documentation for more details.
    negative_slope: float
       Slope of negative side of activations in LeakyReLU
       (when 0 - equivalent to ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple, str] = 1,
        negative_slope: float = 0.01,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Forward pass."""
        return self.layers(x)


def get_custom_backbone():
    """Create small VGG-like model.

    Returns
    -------
    torch.nn.Sequential
        Small sequential model.
    """
    return nn.Sequential(
        BackboneBlock(1, 16, kernel_size=(3, 3), padding=0, stride=1),
        BackboneBlock(16, 32, kernel_size=(3, 3), padding=0, stride=1),
    )


def get_backbone(model_name: str,
                 path: Optional[str] = None,
                 inference_only: Optional[bool] = False):
    """Construct backbone model and restore pretrained state if provided.

    Parameters
    ----------
    model_name : str
        Backbone model name that is supported by this library.
        ['custom', 'resnet-18'].
    path : Optional[str]
        Optional path to saved model states.
    inference_only : bool
        When set to True only reproduces backbone architecture
        without loading pretrained backbone

    Returns
    -------
    torch.nn.Module
        Backbone model.
    """
    if model_name == "custom":
        backbone = get_custom_backbone()
        out_channels = 32
    elif model_name == "resnet-18":
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        resnet = nn.Sequential(*list(resnet.children())[:-2])

        backbone = resnet
        out_channels = 512
    elif model_name.startswith("resnet-50"):
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        out_channels = 2048
        if model_name == "resnet-50/2":
            resnet = nn.Sequential(*list(resnet.children())[:6])
            out_channels = 512
        backbone = resnet
    else:
        raise ValueError("Unimplemented backbone")

    if path is not None and not inference_only:
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        backbone.load_state_dict(state_dict)

    backbone.out_channels = out_channels
    return backbone


class Normalize(nn.Module):
    """Adapted from original implementation of GeneralizedRCNNTransform.
    ref: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py  # noqa
    """

    def __init__(self, image_mean, image_std):
        super(Normalize, self).__init__()
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):  # noqa D104
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self._normalize(image)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self._batch_images(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def _normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(f"Expected input images to be of floating type "
                            f"but found type {image.dtype} instead")
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def _max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _batch_images(self, images, size_divisible=32):
        max_size = self._max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        """Resize predicted boxes to a size that fits the original image."""
        if self.training:
            return result
        for i, (pred, im_s,
                o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = _resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        _indent = "\n    "
        format_string += (
            f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})")
        format_string += "\n)"
        return format_string


def _resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) for s,
        s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
