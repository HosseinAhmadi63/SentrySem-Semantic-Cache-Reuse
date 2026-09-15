# Third-party data and model notice

The MIT License in this repository applies to the original SentrySem source code, configuration files, tests, and project documentation. It does not relicense third-party datasets, pretrained model weights, or source images represented in derived research artifacts.

## CIFAR-10

The experiments use the CIFAR-10 dataset described by Alex Krizhevsky in *Learning Multiple Layers of Features from Tiny Images* (2009). The raw dataset is not included in this repository. The raw-image reproduction command obtains it through TorchVision from the [official CIFAR-10 distribution page](https://www.cs.toronto.edu/~kriz/cifar.html).

The checked-in `frozen_embeddings.npz` file contains derived semantic features. The `qualitative_examples.npz` file contains the twelve CIFAR-10-derived image views required to regenerate Figure 6, and Figure 6 embeds those views. Use and redistribution of these materials remain subject to the rights and terms applicable to CIFAR-10 and its underlying images.

## Pretrained ResNet-18

The semantic encoder uses TorchVision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`. The model checkpoint is not included in this repository; the raw-image reproduction command obtains it through TorchVision. The frozen feature archive contains outputs produced by that encoder. Consult the [TorchVision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and the applicable ImageNet terms before redistributing model weights or derived assets.

## Python dependencies

NumPy, pandas, Matplotlib, Pillow, ReportLab, pypdf, pypdfium2, PyTorch, TorchVision, pytest, pytest-cov, and Ruff are installed as dependencies and are not vendored. Each dependency retains its own license. In particular, pypdfium2 and PDFium are available under permissive Apache-2.0/BSD-style terms; their binary distributions include the applicable notices. Exact supported versions are listed in `pyproject.toml` and the requirements files.
