"""Loss function for the STFPM Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np


class STFPMLoss(nn.Module):
    """Feature Pyramid Loss This class implmenents the feature pyramid loss function proposed in STFPM paper.

    Example:
        >>> from anomalib.models.components.feature_extractors import FeatureExtractor
        >>> from anomalib.models.stfpm.loss import STFPMLoss
        >>> from torchvision.models import resnet18

        >>> layers = ['layer1', 'layer2', 'layer3']
        >>> teacher_model = FeatureExtractor(model=resnet18(pretrained=True), layers=layers)
        >>> student_model = FeatureExtractor(model=resnet18(pretrained=False), layers=layers)
        >>> loss = Loss()

        >>> inp = torch.rand((4, 3, 256, 256))
        >>> teacher_features = teacher_model(inp)
        >>> student_features = student_model(inp)
        >>> loss(student_features, teacher_features)
            tensor(51.2015, grad_fn=<SumBackward0>)
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor, s_imagenet_out: Tensor) -> Tensor:
        """Compute layer loss based on Equation (1) in Section 3.2 of the paper.

        Args:
          teacher_feats (Tensor): Teacher features
          student_feats (Tensor): Student features

        Returns:
          L2 distance between teacher and student features.
        """

        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        layer_loss = (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

        N : dict[str, float]= torch.mean(torch.pow(s_imagenet_out,2))
        
        layer_loss = torch.mean(layer_loss) + 0.3*N
        return layer_loss

    def forward(self, teacher_features:  Tensor, student_features:  Tensor, imagenet_params: dict[str, float]) -> Tensor:
        """Compute the overall loss via the weighted average of the layer losses computed by the cosine similarity.

        Args:
          teacher_features (dict[str, Tensor]): Teacher features
          student_features (dict[str, Tensor]): Student features

        Returns:
          Total loss, which is the weighted average of the layer losses.
        """

        layer_losses: list[Tensor] = []
        for i in range(2):
            loss = self.compute_layer_loss(teacher_features[i+1], student_features[i+1], imagenet_params[i+1])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss