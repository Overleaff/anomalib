"""PyTorch model for the STFPM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from torch import Tensor, nn
import torch

from anomalib.models.components import FeatureExtractor
from anomalib.models.stfpm.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
from anomalib.models.stfpm.imagenet import ImageNetDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from itertools import cycle



class STFPMModel(nn.Module):
    """STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

    Args:
        layers (list[str]): Layers used for feature extraction
        input_size (tuple[int, int]): Input size for the model.
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
    """

    def __init__(
        self,
        layers: list[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = FeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers)
        self.student_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=False, layers=layers, requires_grad=True
        )
        
        self.data_transforms_imagenet = transforms.Compose([ #We obtain an image P ∈ R 3×256×256 from ImageNet by choosing a random image,
                        transforms.Resize((512, 512)), #resizing it to 512 × 512,
                        transforms.RandomGrayscale(p=0.3), #converting it to gray scale with a probability of 0.3
                        transforms.CenterCrop((256,256)), # and cropping the center 256 × 256 pixels
                        transforms.ToTensor(),
                        ])

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        # Create the anomaly heatmap generator whether tiling is set.
        # TODO: Check whether Tiler is properly initialized here.
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size)
        

    def forward(self, images: Tensor) -> Tensor | dict[str, Tensor] | tuple[dict[str, Tensor]]:
        """Forward-pass images into the network.

        During the training mode the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images (Tensor): Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.
        """
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, Tensor] = self.teacher_model(images)
        student_features: dict[str, Tensor] = self.student_model(images)
        
        imagenet = ImageNetDataset(imagenet_dir="/content/tiny-imagenet-200/tiny-imagenet-200/train",transform=self.data_transforms_imagenet)
        imagenet_loader = DataLoader(imagenet,batch_size=32,shuffle=True)
        # len_traindata = len(dataset)
        imagenet_iterator = cycle(iter(imagenet_loader))

        # Choose a random pretraining image P ∈ R 3×256×256 from ImageNet [54]
        image_p = next(imagenet_iterator)
        # pdb.set_trace()
        s_imagenet_out: dict[str, Tensor] = self.student_model(image_p[0].cuda())


        
        
        if self.training:
            output = teacher_features, student_features, s_imagenet_out
        else:
            output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)
            if self.tiler:
                output = self.tiler.untile(output)

        return output
