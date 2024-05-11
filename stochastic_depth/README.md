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

```python
        identity = x.clone()
```

![ResNet Identity](../resnet/images/resnet_identity.png)

Then, we will go either into the training motion or the testing one.
Remember, the whole purpose of the paper is to have a **short** network at training time and a **big** network at testing time.

Therefore, inside the layers we will either shut off the layer with probability of survival `prob` during training or using all the layers during testing.

## StoDepth_BasicBlock | forward | training loop

```python
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

```

In this loop, the author decided to split the action depending if the bernoulli random variable is 1 or 0.

**Note:** This might not be the best way to implement this branch because technically you don't need to have this level of separation. You could theorically structure this motion by always having the bernoulli tensor multiplying the output like in the official pytorch doc:

```python
[docs]def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    #[...]

    survival_rate = 1.0 - p

    noise = torch.empty(size, dtype=input.dtype, device=input.device)

    noise = noise.bernoulli_(survival_rate) # <---------------------------------- HERE
    if survival_rate > 0.0:
        noise.div_(survival_rate) # <----------------------------------------------HERE
    return input * noise # <--------------------------------------------------------AND HERE

```

Anyway, let's keep going.

How we are making the distinction about the two case is with this if statement:
```python
if torch.equal(self.m.sample(),torch.ones(1)):
```
If the `m` tensor is equal to 1 then it's fine, we don't need to skip the layer.
```python
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
```
We do all the normal resnet stuff with the bypassing and such, like in this image:

![Resnet Basic Block](../resnet/images/resnet_figure_2.png)

If we need to bypass then we don't do any of the convolution motion:

```python
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
```
Which basically means, output the `x` that we cloned earlier.

## StoDepth_BasicBlock | forward | testing loop
For the testing loop we have the following:

```python
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
```

Here we do the normal resnet stuff up until this part:
```python
        if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity
```
In the original paper we have this formula for the testing phase:

![Stochastic Depth Test Formula](./images/stochastic_depth_formula_test.png)

Which basically says that we should multiply the probability of survival with the output of the convolutions and then add identiy.

However, the author of the repository we took this code from saw that doing so might deteriorate the performance.
Therfore they give you a way to use that method or simply use the normal resnet formula that don't make use of the probability of survival of a layer.

And that' it for the forward and BasicBlock class!

Let's check out now the Stochastic Depth Resnet class:
## ResNet_StoDepth_lineardecay

```python
class ResNet_StoDepth_lineardecay(nn.Module):

    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_StoDepth_lineardecay, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0]-prob_0_L[1]
        self.prob_step = self.prob_delta/(sum(layers)-1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_lineardecayBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_lineardecayBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```

There is three functions of interest here:
- `__init__`: constructor for the initialization
- `_make_layers`: create the right layers with the proper configurations.
- `forward`: that is use to move the input through the layers

Let's take a look at `forward` first:

## ResNet_StoDepth_lineardecay | forward
```python
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```
This has the same flow that in the ResNet paper (because this is a resnet afterall).
We are doing the motion highlighted in this table:

![Resnet Table 1](../resnet/images/resnet_table_1.png)

Nothing specific to the stochastic depth here.


## ResNet_StoDepth_lineardecay | _make_layers
```python
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample)) # < -- !
        self.prob_now = self.prob_now - self.prob_step # < -- !
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes)) # < -- !
            self.prob_now = self.prob_now - self.prob_step # < -- !

        return nn.Sequential(*layers)

```
This functions looks very similar than the one found in the Pytorch documentation for resnet, since it's the same kind of flow.

The one difference is at these specific point highlighted above!
```python
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample)) # < -- !
        self.prob_now = self.prob_now - self.prob_step # < -- !

        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes)) # < -- !
            self.prob_now = self.prob_now - self.prob_step # < -- !

```
What is happening in a nutshell is that we are creating the specific `StoDepth_BasicBlock`. This specific block is not in the for loop because it's the one block where downsampling might need to happen in the layer. 
```python
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample)) # < -- !
```
Then we are substracting to the survival probability to implement the linear decay from the paper:
```python
        self.prob_now = self.prob_now - self.prob_step # < -- !
```

![Linear Decay](images/stochastic_depth_linear_decay.png)

Finally we repeat the process multiple time without downsampling:
```python
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes)) # < -- !
            self.prob_now = self.prob_now - self.prob_step # < -- !
```

We will make use of the `_make_layers` function in the constructor

## ResNet_StoDepth_lineardecay | _init_
```python
    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_StoDepth_lineardecay, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0]-prob_0_L[1]
        self.prob_step = self.prob_delta/(sum(layers)-1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_lineardecayBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_lineardecayBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

```

There are three section in this function:
- Parameter initialization
- Layers construction
- Layers Initiatlization


**Parameter Initialization**
```python
        super(ResNet_StoDepth_lineardecay, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0]-prob_0_L[1]
        self.prob_step = self.prob_delta/(sum(layers)-1)

```
Nothing too complicate here, we are initializing the different operations being used and finnally setting up the steps we will be using for the linear decay.

**Layers Construction**
```python
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

```
We are using repeatedly the `_make_layer` function we've already discussed above and we are finishing up with the fully connected layer for the classification.

**Layers Initialization**
```python
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_lineardecayBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_lineardecayBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
```

A whole bunch of layer specific initialization that fit the best. This doesn't have to do necessarily with the stochastic depth operations.