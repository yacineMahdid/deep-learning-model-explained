# Stochastic Depth
We'll be breaking down the code from this [repository](https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet/blob/master/TYY_stodepth_lineardecay.py) which makes use of linear decay stochastic depth.

Note: I've trimmed down the whole library so that we can solely focus on the overall structure, we'll be exploring resnet 18.

## resnet18_StoDepth_lineardecay
Top level function to call to create the model as follow
```python
net = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,0.5], multFlag=True) 
```

The source code is like this:
```python
def resnet18_StoDepth_lineardecay(pretrained=False, prob_0_L=[1,0.5], multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth_lineardecay-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth_lineardecay(StoDepth_BasicBlock, prob_0_L, multFlag, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
```

This isn't an official Pytorch implementation so the nomenclature are a bit different than usual, here is what the parameters means:
- `pretrained`: define if we need to load the model weight already pre-trained.
- `prob_0_L`: this is two numbers which are the starting  and the end survival probability for a given layer (note: should have been two parameters).
- `multFlag`: parameter to do or not the multiplication at test time.

Within this function the interesting bit for us is this one:
```python
    model = ResNet_StoDepth_lineardecay(StoDepth_BasicBlock, prob_0_L, multFlag, [2, 2, 2, 2], **kwargs)
```

Before diving into this function, we'll take a look at how the basic block of the ResNet hass been modified to match the residual depth procedure.

## StoDepth_BasicBlock

```python

class StoDepth_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):
        
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out

```

There is two function of interest here, the initialization and the forward function.
Let's look at each of them.

## StoDepth_BasicBlock | __init__

```python
    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag
```

This is fairly standard initialization, except for this tensor:
```python
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
```
This obscure variable is extremly important as its the one that will define whether the layer will get bypassed or not.
We are using a bernoulli distribution to have an output that is either 0 (bypassed) or 1 (not bypassed) following a probability `prob`.

Let's use these variables in the `forward` function

## StoDepth_BasicBlock | forward

```python
    def forward(self, x):
        
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out
```

The first step is to modify the input x so that we can use it as the `identity` element of the resnet:

![ResNet Identity](../resnet/images/resnet_identity.png)