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

## cfgs

```python
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
```
This is simply a way for the configuration of the networks to be expressed.
Here M means a max pooling layer and the integers are used for convolution layers input channels.


let's check out the make layers function next.
## make_layers
```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

The make layers consist of a main iteration on the configuration being given, as a reminder the cfg is a dictionary that give access to list of str or int. For instance, A gives us:

```python
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
```
Meaning the variable v will iterate on `[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]`.

This is a bit of a weird way of building the network considering how usually the Pytorch documentation is set up, but it works.

First we have in_channel which starts at 3, this will change as we iterate across the list and be fed for the convolution layer.

Then we have the `"M"` variable which in this case will trigger:

```python
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
```
Which creates a 2x2 max pooling layer of stride 2 and add it to the bunch of layers.

Next we have all the numbers `[64, 128, 256, 512]`, which goes into the else statement.

```python
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
```

We create a convolution layer, then we either add a batchnorm + relu or just a relu.

Finally we return the layers as a sequential object:
```python
    return nn.Sequential(*layers)
```

Note: The final classifier layers aren't in there, it will be located in the VGG class.

Pretty simple. Let's take a look at VGG



## VGG
```python
class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

The network main input is the output of make layers, technically it would be a bit cleaner to add the make layer **inside** the constructor which is usually what is being done in newer models.

So the input is Sequential which is a series of layer before the final classifier.

Here we have the classic 3 section:
1. forward function
2. weights initializer
3. layer creation

## VGG | init -> layer creation

For the first chunk we have:
```python
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
```

The parameters of interest are:
- `features` : the output of make_layers
- `num_class` : the number of class for the final classification
- `init_weights` : whether we will do the recommended weight initialization or not
- `dropout`: finally the amount of drop out involved in the last classification layers (aka whether some of the neuron will die or not).


The final classifier is:
- Linear + Relu with dropp out
- another linear + relu + dropp out
- final linear with the output being the required number of class (i.e. 1000 for imagenet)

## VGG | init > weight initialization
```python
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

```
Here we are iterating over all layers and initializing their weight and/or bias with a specific initialization scheme.

Lets finally look at the forward function:

## VGG | forward
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

The forward function accet the input image x and then pass it through the whole architecture in this order:
1. `features`: aka all the layers before the final classifier
2. `avgpool` : aka average pooling before doing the final classification
3. `flattening` : aka we are flattening the output of the average pooling to have like 1 dimension instead of being 3D.
4. `classification` : aka shoving this flattened output into the final fully connected classification units.

and that's it 🎉