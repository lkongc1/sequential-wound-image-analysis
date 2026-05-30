"""Model factory for creating wound segmentation models."""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

# Runtime import with fallback
try:
    import segmentation_models_pytorch as smp  # type: ignore[reportMissingImports]
    _SMP_AVAILABLE = True
except ImportError:
    smp = None  # type: ignore[assignment]
    _SMP_AVAILABLE = False


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator that registers a model factory under *name*.

    Parameters
    ----------
    name:
        Key used later with :func:`create_model`.

    Returns
    -------
    Decorator that stores the callable in :data:`MODEL_REGISTRY`.
    """

    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def create_model(name: str, **kwargs: object) -> nn.Module:
    """Factory function to create segmentation models (OCP - Open/Closed Principle).

    Args:
        name: Registered model name (e.g. ``"unet"``, ``"attention_unet"``).
        **kwargs: Forwarded to the registered factory callable.

    Returns:
        model: nn.Module instance

    Raises:
        ValueError: if *name* is not registered in :data:`MODEL_REGISTRY`.
    """
    factory = MODEL_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type: '{name}'. "
            f"Available models: {available}"
        )
    return factory(**kwargs)


# ------------------------------------------------------------------ #
# SMP model wrappers (registered here to avoid editing factory later)
# ------------------------------------------------------------------ #


@register_model("unet")
def _create_unet(
    encoder_name: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    assert smp is not None
    encoder_weights = "imagenet" if pretrained else None
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


@register_model("deeplabv3")
def _create_deeplabv3(
    encoder_name: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    assert smp is not None
    encoder_weights = "imagenet" if pretrained else None
    return smp.DeepLabV3(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


@register_model("manet")
def _create_manet(
    encoder_name: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    assert smp is not None
    encoder_weights = "imagenet" if pretrained else None
    return smp.MAnet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


@register_model("fpn")
def _create_fpn(
    encoder_name: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    assert smp is not None
    encoder_weights = "imagenet" if pretrained else None
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


@register_model("deeplabv3plus")
def _create_deeplabv3plus(
    encoder_name: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch not installed. "
            "Run: pip install segmentation-models-pytorch"
        )
    assert smp is not None
    encoder_weights = "imagenet" if pretrained else None
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


# ------------------------------------------------------------------ #
# Import custom model modules to trigger @register_model side effects
# ------------------------------------------------------------------ #
from src.models import unet, attention_unet, nested_unet  # noqa: E402, F401, I001
