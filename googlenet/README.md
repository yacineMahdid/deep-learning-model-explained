# GoogleNet
The code is a modified version of the official [googlenet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py) on Pytorch.

This code is fairly verbose, but relatively clean, let's go top API to bottom internal.

## googlenet
```python
@register_model()
@handle_legacy_interface(weights=("pretrained", GoogLeNet_Weights.IMAGENET1K_V1))
def googlenet(*, weights: Optional[GoogLeNet_Weights] = None, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        weights (:class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.GoogLeNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.GoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
    """
    weights = GoogLeNet_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GoogLeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model
```

We start with the above which is fairly standard.

The part that matter the most is this one:
```python
    model = GoogLeNet(**kwargs)
```
Which simply call GoogLeNet class to create the model and shove all of the parameters. The rest is mostly just to load in the  weights.

Before getting into GoogLeNet class, let's explore the Inception module.

## Inception


Now that we have an understanding of how Inception works, let's take a look at the auxiliary classification head called `InceptionAux`.

## InceptionAux

BasicConv2D is fairly straightforward, let's take a look at it.

## BasicConv2D

Okay now that we have all the pieces, let's put it all together on the GoogLeNet class!

## GoogLeNet

And voila 🎉!