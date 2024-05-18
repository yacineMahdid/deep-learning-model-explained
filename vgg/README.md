# VGG Network

Code is taken from the official Pytorch [repository](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py) in the torchvision models section.


Code has been broken down for the sole purpose of education, it won't be workeable.

## vgg19 upper most API call
The top most way to create a VGG model, like the VGG 19 with batch normalization is as follow:

```python
model = vgg19_bn()
```

Let's break it down one layer.


```python
@register_model()
@handle_legacy_interface(weights=("pretrained", VGG19_BN_Weights.IMAGENET1K_V1))
def vgg19_bn(*, weights: Optional[VGG19_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, weights, progress, **kwargs)
```

Cutting the non-essential part for understanding the architecture out of the above code, we get:

```python
def vgg19_bn(weights, progress):
    return _vgg("E", True, weights, progress, **kwargs)
```

Here it calls the `_vgg` internal function with a bunch of parameters of interest. Let's investigate this function next.

## _vgg
```python
def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model
```

Here there are a bunch of parameters of importance let's break them down:
1. cfg : string that signify what flavor of VGG we will be using denoted by a letter (A,B,D,E from the paper are supported).
2. batch_norm: boolean that says whether we are going to use batch normalization or not.
3. weights: object which represents the weights for the pre-trained model.

The section of interest here is the following:
```python
def _vgg(cfg, batch_norm, weights):
    model = VGG(make_layers(cfgs[cfg],batch_norm))
    return model
```

Let's break it down some more in term of important section:
1. `VGG` is the class that will create the model
2. `make_layers` is a function that is supposed to create all the layers and will feed that as params for VGG.
3. `cfgs[cfg]` will return the exact configuration which maps to the letts

let's take a look at cfgs[cfg] first

