# DenseNet
From the top we have the following API for the [densenet121 variant](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.densenet121):
```python
torchvision.models.densenet121(*, weights: Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs: Any)
```

## densenet121
Digging inside the torchvision library we find the following:
```python
@register_model()
@handle_legacy_interface(weights=("pretrained", DenseNet121_Weights.IMAGENET1K_V1))
def densenet121(*, weights: Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet121_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet121_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet121_Weights
        :members:
    """
    weights = DenseNet121_Weights.verify(weights)

    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)
```

Removing the pytorch internal information we get:

```python
def densenet121(*, weights, progress, **kwargs):
    weights = DenseNet121_Weights.verify(weights)
    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)
```

What's actually important here is the following line:
```python
_densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)
```
Which is the internal representation of densenet.

# _densenet

The code for the internal densenet function is also straightforward as it's only purpose is calling the DenseNet class with the right parameters.
```python
def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model
```