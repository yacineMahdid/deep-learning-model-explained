# Fractal Net
The code was taken and modified from the [following repository](https://github.com/khanrc/pt.fractalnet/tree/master).

The model is used in the following manner at a high level:
```python
    model = FractalNet(data_shape, config.columns, config.init_channels,
                       p_ldrop=config.p_ldrop, dropout_probs=config.dropout_probs,
                       gdrop_ratio=config.gdrop_ratio, gap=config.gap,
                       init=config.init, pad_type=config.pad, doubling=config.doubling,
                       dropout_pos=config.dropout_pos, consist_gdrop=config.consist_gdrop)
    model = model.to(device)
```

Lots of parameters and with a structure that doesn't necessarily follow the usual Pytorch documentation, which is totally fine.

If we look at a high level overview we'll have the following class nested in this manner:
- FractalNet -> FractalBlock -> ConvBlock

As a disclaimer before we jump into the main code, the FractalNet architecture is a bit weird in the sense that it is not linear.
There are branching paths within a FractalBlock and there isn't the same amount of element within a given columns.

Therefore, programmatically we'll need to work with a grid that we will populate with information that will help flag which region of the grid has something in a given path.

Like the following image (will come up again below):

![Fractal Block](images/fractal_block.png)

**Note about Drop Path:** 
An important thing to keep in mind with this code is that the drop path is done on a per sample basis, not on a per batch basis.

Meaning that drop path will be applied within a batch and cut the sample into some % doing global drop path and the rest will be local drop path.

Let's look at each of the element starting with FractalNet:

## FractalNet
```python
class FractalNet(nn.Module):
    def __init__(self, data_shape, n_columns, init_channels, p_ldrop, dropout_probs,
                 gdrop_ratio, gap=0, init='xavier', pad_type='zero', doubling=False,
                 consist_gdrop=True, dropout_pos='CDBR'):
        """ FractalNet
        Args:
            - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
            - n_columns: the number of columns
            - init_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - gap: pooling type for last block
            - init: initializer type
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - consist_gdrop
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()
        assert dropout_pos in ['CDBR', 'CBRD', 'FD']

        self.B = len(dropout_probs) # the number of blocks
        self.consist_gdrop = consist_gdrop
        self.gdrop_ratio = gdrop_ratio
        self.n_columns = n_columns
        C_in, H, W, n_classes = data_shape

        assert H == W
        size = H

        layers = nn.ModuleList()
        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = FractalBlock(n_columns, C_in, C_out, p_ldrop, p_dropout,
                              pad_type=pad_type, doubling=doubling, dropout_pos=dropout_pos)
            layers.append(fb)
            if gap == 0 or b < self.B-1:
                # Originally, every pool is max-pool in the paper (No GAP).
                layers.append(nn.MaxPool2d(2))
            elif gap == 1:
                # last layer and gap == 1
                layers.append(nn.AdaptiveAvgPool2d(1)) # average pooling

            size //= 2
            total_layers += fb.max_depth
            C_in = C_out
            if b < self.B-2:
                C_out *= 2 # doubling except for last block

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))

        if gap == 2:
            layers.append(nn.Conv2d(C_out, n_classes, 1, padding=0)) # 1x1 conv
            layers.append(nn.AdaptiveAvgPool2d(1)) # gap
            layers.append(Flatten())
        else:
            layers.append(Flatten())
            layers.append(nn.Linear(C_out * size * size, n_classes)) # fc layer

        self.layers = layers

        # initialization
        if init != 'torch':
            initialize_ = {
                'xavier': nn.init.xavier_uniform_,
                'he': nn.init.kaiming_uniform_
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1: # weights only
                    initialize_(p)
                else: # bn w/b or bias
                    if 'bn.weight' in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, x, deepest=False):
        if deepest:
            assert self.training is False
        GB = int(x.size(0) * self.gdrop_ratio)
        out = x
        global_cols = None
        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                if not self.consist_gdrop or global_cols is None:
                    global_cols = np.random.randint(0, self.n_columns, size=[GB])

                out = layer(out, global_cols, deepest=deepest)
            else:
                out = layer(out)

        return out
```

```
Parameters are:
    - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
    - n_columns: the number of columns
    - init_channels: the number of out channels in the first block
    - p_ldrop: local drop prob **(!!! this is for local drop path)**
    - dropout_probs: dropout probs (list) **(!!! This is for regular drop out)**
    - gdrop_ratio: global droppath ratio **(!!! This is for global drop path)**
    - consist_gdrop
    - dropout_pos: the position of dropout
        - CDBR (default): conv-dropout-BN-relu
        - CBRD: conv-BN-relu-dropout
        - FD: fractal_block-dropout

    (less important parameters)
    - gap: pooling type for last block
    - init: initializer type
    - pad_type: padding type of conv
    - doubling: if True, doubling by 1x1 conv in front of the block.
```
We'll see most of these reappering multiple time throughout the downstream code.

Let's take a look at the constructor first:

### FractalNet | init
```python
def __init__(self, data_shape, n_columns, init_channels, p_ldrop, dropout_probs,
                 gdrop_ratio, gap=0, init='xavier', pad_type='zero', doubling=False,
                 consist_gdrop=True, dropout_pos='CDBR'):
        """ FractalNet
        Args:
            - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
            - n_columns: the number of columns
            - init_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - gap: pooling type for last block
            - init: initializer type
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - consist_gdrop
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()
        assert dropout_pos in ['CDBR', 'CBRD', 'FD']

        self.B = len(dropout_probs) # the number of blocks
        self.consist_gdrop = consist_gdrop
        self.gdrop_ratio = gdrop_ratio
        self.n_columns = n_columns
        C_in, H, W, n_classes = data_shape

        assert H == W
        size = H

        layers = nn.ModuleList()
        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = FractalBlock(n_columns, C_in, C_out, p_ldrop, p_dropout,
                              pad_type=pad_type, doubling=doubling, dropout_pos=dropout_pos)
            layers.append(fb)
            if gap == 0 or b < self.B-1:
                # Originally, every pool is max-pool in the paper (No GAP).
                layers.append(nn.MaxPool2d(2))
            elif gap == 1:
                # last layer and gap == 1
                layers.append(nn.AdaptiveAvgPool2d(1)) # average pooling

            size //= 2
            total_layers += fb.max_depth
            C_in = C_out
            if b < self.B-2:
                C_out *= 2 # doubling except for last block

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))

        if gap == 2:
            layers.append(nn.Conv2d(C_out, n_classes, 1, padding=0)) # 1x1 conv
            layers.append(nn.AdaptiveAvgPool2d(1)) # gap
            layers.append(Flatten())
        else:
            layers.append(Flatten())
            layers.append(nn.Linear(C_out * size * size, n_classes)) # fc layer

        self.layers = layers

        # initialization
        if init != 'torch':
            initialize_ = {
                'xavier': nn.init.xavier_uniform_,
                'he': nn.init.kaiming_uniform_
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1: # weights only
                    initialize_(p)
                else: # bn w/b or bias
                    if 'bn.weight' in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)
```
There are three main section:
1. parameters initialization
2. layer creations
3. layer initialization

The first one is fairly trivial, so let's jump directly to layer creation

```python
        layers = nn.ModuleList()
        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = FractalBlock(n_columns, C_in, C_out, p_ldrop, p_dropout,
                              pad_type=pad_type, doubling=doubling, dropout_pos=dropout_pos)
            layers.append(fb)
            if gap == 0 or b < self.B-1:
                # Originally, every pool is max-pool in the paper (No GAP).
                layers.append(nn.MaxPool2d(2))
            elif gap == 1:
                # last layer and gap == 1
                layers.append(nn.AdaptiveAvgPool2d(1)) # average pooling

            size //= 2
            total_layers += fb.max_depth
            C_in = C_out
            if b < self.B-2:
                C_out *= 2 # doubling except for last block

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))

        if gap == 2:
            layers.append(nn.Conv2d(C_out, n_classes, 1, padding=0)) # 1x1 conv
            layers.append(nn.AdaptiveAvgPool2d(1)) # gap
            layers.append(Flatten())
        else:
            layers.append(Flatten())
            layers.append(nn.Linear(C_out * size * size, n_classes)) # fc layer

        self.layers = layers
```

There is a couple of weird paradigm in this code throughout, like I mentioned earlier this isn't Pytorch official documentation level of clarity.

First off we will create the FractalBlock followed by the pooling layers (we're iterating on the dropout_probs to know how many blocks)

The above could be summarize as:
```python
        for block in blocks:
            fractal_block = FractalBlock(RIGHT_PARAMETERS)
            layers.append(fractal_block)
            layers.append(nn.RIGHT_POOLING_LAYER)
```

Once we are done with creating the fractal block, we create the prediction layer at the end:
```python
        if gap == 2:
            layers.append(nn.Conv2d(C_out, n_classes, 1, padding=0)) # 1x1 conv
            layers.append(nn.AdaptiveAvgPool2d(1)) # gap
            layers.append(Flatten())
        else:
            layers.append(Flatten())
            layers.append(nn.Linear(C_out * size * size, n_classes)) # fc layer

        self.layers = layers
```
Here there is two flavor, a 1x1 convolution or a linear fully connected layer (both are somewhat equivalent).


Then we have the initialization section:
```python
        # initialization
        if init != 'torch':
            initialize_ = {
                'xavier': nn.init.xavier_uniform_,
                'he': nn.init.kaiming_uniform_
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1: # weights only
                    initialize_(p)
                else: # bn w/b or bias
                    if 'bn.weight' in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)
```
Altought the structure is pretty weird here, we do basic inialization on a layer by layer basis.

Let's take a look at the forward function now:

### FractalNet | forward
```python
    def forward(self, x, deepest=False):
        if deepest:
            assert self.training is False

        GB = int(x.size(0) * self.gdrop_ratio)
        out = x

        global_cols = None
        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                if not self.consist_gdrop or global_cols is None:
                    global_cols = np.random.randint(0, self.n_columns, size=[GB])

                out = layer(out, global_cols, deepest=deepest)
            else:
                out = layer(out)

        return out
```

The first thing of importance here is this variable:
```python
        GB = int(x.size(0) * self.gdrop_ratio)
```
Which can be understood as the Global_Batch (GB). It's the size of the batch that global_drop_path should drop.
Local path will be the remainder of that batch in the fractal block.

Then we iterate layer by layer, if we aren't in a fractal block it's fine we just put the out inside the layer (pooling or prediction).

However, if it's a fractalblock layer we will do the following:
```python
    global_cols = np.random.randint(0, self.n_columns, size=[GB])
    out = layer(out, global_cols, deepest=deepest)
```
Note: Here I'm disregarding the `self.consist_gdrop` variable as it's use only when the global_drop_path is consistant throughout and where we only take 1 global path out.

Anyway, the first line will select 1 column index for **per sample** that we are part of the Global_Batch.
The second line will take this array of global drop path column along with the input and push it forward.

And that's basically it, the bulk of the work will happen in the FractalBlock.
However, it's important to keep in mind the few variables that appears in FractalNet.

Let's explore the FractalBlock now.

## FractalBlock
A fractal block is a repeating block containing the different columnar path inside a fractalnet.

```python
class FractalBlock(nn.Module):
    def __init__(self, n_columns, C_in, C_out, p_ldrop, p_dropout, pad_type='zero',
                 doubling=False, dropout_pos='CDBR'):
        """ Fractal block
        Args:
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()

        self.n_columns = n_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        if dropout_pos == 'FD' and p_dropout > 0.:
            self.dropout = nn.Dropout2d(p=p_dropout)
            p_dropout = 0.
        else:
            self.dropout = None

        if doubling:
            #self.doubler = nn.Conv2d(C_in, C_out, 1, padding=0)
            self.doubler = ConvBlock(C_in, C_out, 1, padding=0)
        else:
            self.doubler = None

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns-1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i+1) % dist == 0:
                    first_block = (i+1 == dist) # first block in this column
                    if first_block and not doubling:
                        # if doubling, always input channel size is C_out.
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ConvBlock(cur_C_in, C_out, dropout=p_dropout, pad_type=pad_type,
                                       dropout_pos=dropout_pos)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

    def drop_mask(self, B, global_cols, n_cols):
        """ Generate drop mask; [n_cols, B].
        1) generate global masks
        2) generate local masks
        3) resurrect random path in all-dead column
        4) concat global and local masks

        Args:
            - B: batch_size
            - global_cols: global columns which to alive [GB]
            - n_cols: the number of columns of mask
        """
        # global drop mask
        GB = global_cols.shape[0]
        # calc gdrop cols / samples
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]
        # gen gdrop mask
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.

        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.-self.p_ldrop, [n_cols, LB]).astype(np.float32)
        alive_count = ldrop_mask.sum(axis=0)
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices] = 1.

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    def join(self, outs, global_cols):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        n_cols = len(outs)
        out = torch.stack(outs) # [n_cols, B, C, H, W]

        if self.training:
            mask = self.drop_mask(out.size(1), global_cols, n_cols).to(out.device) # [n_cols, B]
            mask = mask.view(*mask.size(), 1, 1, 1) # unsqueeze to [n_cols, B, 1, 1, 1]
            n_alive = mask.sum(dim=0) # [B, 1, 1, 1]
            masked_out = out * mask # [n_cols, B, C, H, W]
            n_alive[n_alive == 0.] = 1. # all-dead cases
            out = masked_out.sum(dim=0) / n_alive # [B, C, H, W] / [B, 1, 1, 1]
        else:
            out = out.mean(dim=0) # no drop

        return out

    def forward(self, x, global_cols, deepest=False):
        """
        global_cols works only in training mode.
        """
        out = self.doubler(x) if self.doubler else x
        outs = [out] * self.n_columns
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = [] # outs of current depth
            if deepest:
                st = self.n_columns - 1 # last column only

            for c in range(st, self.n_columns):
                cur_in = outs[c] # current input
                cur_module = self.columns[c][i] # current module
                cur_outs.append(cur_module(cur_in))

            # join
            #print("join in depth = {}, # of in_join = {}".format(i, len(cur_out)))
            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined

        if self.dropout_pos == 'FD' and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1] # for deepest case
```
There are a few section of interest here, namely:
- constructor (init)
- drop mask function
- join function
- forward function

Let's look at all of them, starting with the constructor

### FractalBlock | init
```python
    def __init__(self, n_columns, C_in, C_out, p_ldrop, p_dropout, pad_type='zero',doubling=False, dropout_pos='CDBR'):
        """ Fractal block
        Args:
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()

        self.n_columns = n_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        if dropout_pos == 'FD' and p_dropout > 0.:
            self.dropout = nn.Dropout2d(p=p_dropout)
            p_dropout = 0.
        else:
            self.dropout = None

        if doubling:
            #self.doubler = nn.Conv2d(C_in, C_out, 1, padding=0)
            self.doubler = ConvBlock(C_in, C_out, 1, padding=0)
        else:
            self.doubler = None

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns-1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i+1) % dist == 0:
                    first_block = (i+1 == dist) # first block in this column
                    if first_block and not doubling:
                        # if doubling, always input channel size is C_out.
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ConvBlock(cur_C_in, C_out, dropout=p_dropout, pad_type=pad_type,
                                       dropout_pos=dropout_pos)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2
```


As a note, doubling in this text refer to this easy to miss footnote in the paper:
> This deeper (4 column) FractalNet has fewer parameters. We vary column width: p128, 64, 32, 16q channels across columns initially, doubling each block except the last. A linear projection temporarily widens thinner columns before joins. As in Iandola et al. (2016), we switch to a mix of 1 ˆ 1 and 3 ˆ 3 convolutional filters.

Let's keep that in mind while we explore the constructor.

Within the constructor there is a few area of interest:
1. various initialization
2. drop out preparation
3. creation of the columns

Let's break it down some more:
#### FractalBlock | Init | Initialization & dropout
```python
        self.n_columns = n_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        self.dropout = nn.Dropout2d(p=p_dropout)
        self.doubler = ConvBlock(C_in, C_out, 1, padding=0)
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns-1)
        self.count = np.zeros([self.max_depth], dtype=np.int)
```
Also there are these inputs given to the constructor:
```python
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
```

The normal stuff in there are the following:
1. `n_columns` : this simply states how many columns we should be creating, in the paper it was called `C`
2. `C_in and C_out` : the channel in and out to start the fractal block
3. `p_dropout` : the probability of a drop out (not drop path do not get confused)
4. `p_ldrop` : the probability of a local drop path, which is happening at each joins.
5. `pad_type` : variable sed in the ConvBlock for the padding type.
6. `doubling` : boolean used to figure out if we are adding a 1x1 convolution, it pertain to the note above.
7. `dropout_pos`: the position of the drop out, here at this level only FD can be used, the other two will be passed as parameter for the ConvBlock

There are a few weird variables that aren't so well documented in there which are the following:
```python
        self.doubler = ConvBlock(C_in, C_out, 1, padding=0)
        self.max_depth = 2 ** (n_columns-1)
        self.count = np.zeros([self.max_depth], dtype=np.int)
```
- The doubler is the 1x1 convolution that we add in very specific case.
- max_depth is the formula 2^(C-1) that tells us the max depth in a block, it depends on C which is the number of columns.
- count is an array where the index represent the depth in the fractal block. This will be used to count the number of convolution block we have per depth or rows (in turn useful for the join operation)

for drop out we are simply doing it if it's FD block:
```python
        if dropout_pos == 'FD' and p_dropout > 0.:
            self.dropout = nn.Dropout2d(p=p_dropout)
            p_dropout = 0.
        else:
            self.dropout = None
```
Notice here that we aren't going to add that to our layer list. self.dropout however will make a comeback in the forward function.

#### FractalBlock | Init | Columns Creation
```python
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)

        for col in self.columns:
            for i in range(self.max_depth):
                if (i+1) % dist == 0:
                    first_block = (i+1 == dist) # first block in this column
                    if first_block and not doubling:
                        # if doubling, always input channel size is C_out.
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ConvBlock(cur_C_in, C_out, dropout=p_dropout, pad_type=pad_type,
                                       dropout_pos=dropout_pos)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2
```
In this section we are effectively populating two arrays:
1. columns, which is a 2D array of ModuleList, basically a grid representing every row and column in a fractalblock
2. count, an array that keep a count of how many convolutional blocks there are per level (will be useful for the join function)

Then the recipe is as follow:
- We are iterating on each column one by one across its whole depth.
- by default we will output a `None` inside our column array, except when this expression is true : `if (i+1) % dist == 0`

This expression needs to be taken into consideration with how the variable `dist` was initialized and how it is updated:
```python
        dist = self.max_depth
        # FOR LOOP STUFF
            if (i+1) % dist == 0
                # ADD A MODULE
            else:
                # DONT ADD A MODULE
        
        dist //= 2
```
In the first column, we will only add a convolutional block when i+1 == dist which is at the max depth (so once)
In the second column, we will add a convolutional block twice since we halve the dist variable, so we'll hit the modulus of it 2 time.
In the subsequent column we will do it 2* the previous column.
Until we hit the last column where we will add a convolutional block all the time instead of the None.

There are some more details about the doubling in there, but for the overall architecture it isn't too important.

So, at the end of these iterations we should have a columns grid filled up with None of Convolutional filters.
A counts array per row that sum the amount of convolutional filters.

Let's jump down to the forward function, which is crucial to understand how we are operating the joins and drop path.

### FractalBlock | forward
```python
    def forward(self, x, global_cols, deepest=False):
        """
        global_cols works only in training mode.
        """
        out = self.doubler(x) if self.doubler else x
        outs = [out] * self.n_columns
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = [] # outs of current depth
            if deepest:
                st = self.n_columns - 1 # last column only

            for c in range(st, self.n_columns):
                cur_in = outs[c] # current input
                cur_module = self.columns[c][i] # current module
                cur_outs.append(cur_module(cur_in))

            # join
            #print("join in depth = {}, # of in_join = {}".format(i, len(cur_out)))
            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined

        if self.dropout_pos == 'FD' and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1] # for deepest case
```
Notice here that we have our global_cols from FractalNet that made a comeback.

In this architecture, we will be feeding the inputs in lockstep across all columns, there are some very tricky flow here but we'll go through it.

First:
```python
        outs = [out] * self.n_columns
```
We are multiplying x (which is equal to out) n_columns time before jumping into the iterations. Effectively feeding each of the columns the initial input.
From now on the input to the fractal block will be an array.

At the end we have:
```python
        if self.dropout_pos == 'FD' and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1] # for deepest case
```
Where it's basically drop out applied at the end completly of the depth.

The only thing we need to return is the last outs we have at the deepest level of our fractalBlock.

In between is the real work:
```python
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = [] # outs of current depth
            if deepest:
                st = self.n_columns - 1 # last column only

            for c in range(st, self.n_columns):
                cur_in = outs[c] # current input
                cur_module = self.columns[c][i] # current module
                cur_outs.append(cur_module(cur_in))

            # join
            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined
```

The first variable is pretty important:
```python
st = self.n_columns - self.count[i]
```
No idea what `st` stand for, but in a nutshell by doing `n_columns - count[depth]` we are saying where we should be doing the join at which index.
The image in the original paper is a bit confusing and stylized, because the grid is more clearer like this:

![Fractal Block](images/fractal_block.png)

So at the first iteration in this example at the highest depth, we'll have:
`st = 4 - 1 = 3`

we'll then do
```python
for c in range(3, 4): #AKA ONE ITERATION
    cur_in = outs[3] # current input
    cur_module = self.columns[3][0] # current module
    cur_outs.append(cur_module(cur_in))

joined = self.join(cur_outs, global_cols)

for c in range(3, 4): #AKA ONE ITERATION
    outs[3] = joined
```

So this simplify to:
```python
cur_in = outs[3] # current input
cur_module = self.columns[3][0] # current module
cur_outs.append(cur_module(cur_in))
joined = self.join(cur_outs, global_cols)
outs[3] = joined
```

- We take the input out of our array of output
- we get the right module out of our grid of module (will not be None) thanks to `st` variable
- we jam the input into the layer and append the output to that to our array of current output (a singular element array in this example)
- then we call join with the global_cols variable (will check in a few)
- The output of that is that added as the outs for the next iteration.

In the first example it's trivial since there is no iterations, but we get the idea.

Something very important to note in the drawing is that **There is always a join happening at all depth**. They haven't drawn it in the image because some of the join are happening on a singular convolutional element. The element wise mean of one element is the element.

Therefore, in the code, this is why you see the join happening at all time, it doesn't change anything if it's for a singular element.

### FractalBlock | join
```python
    def join(self, outs, global_cols):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        n_cols = len(outs)
        out = torch.stack(outs) # [n_cols, B, C, H, W]

        if self.training:
            mask = self.drop_mask(out.size(1), global_cols, n_cols).to(out.device) # [n_cols, B]
            mask = mask.view(*mask.size(), 1, 1, 1) # unsqueeze to [n_cols, B, 1, 1, 1]
            n_alive = mask.sum(dim=0) # [B, 1, 1, 1]
            masked_out = out * mask # [n_cols, B, C, H, W]
            n_alive[n_alive == 0.] = 1. # all-dead cases
            out = masked_out.sum(dim=0) / n_alive # [B, C, H, W] / [B, 1, 1, 1]
        else:
            out = out.mean(dim=0) # no drop

        return out
```
The parameters of interest are:
- `outs`: the input to join which is the output of the previous fractal layers manipulation
- `global_cols`: parameter for drop_mask which dictate for the global path which columns to drop. This parameters is passed along all the way from the FractalNet class we'll see in a bit.

Here there are two main case, if we are not in training mode we will calculate the mean with no drop and generate the output for the next layer.

However, if we are in training we will follow this sequence:
1. create the mask of column to drop out using the `drop_mask` function
2. calculating how much column are still alive.
3. masking the inputs
4. calculating the means for this join operation.

It's a bit confusing here with the alive/dead nomenclature, but this part:
`n_alive[n_alive == 0.] = 1` is only to not make the division carry on zero in the denominator.

Let's dive into drop_mask now.

### FractalBlock | drop_mask
drop_mask is used as follows by the join operation:
```python
mask = self.drop_mask(out.size(1), global_cols, n_cols).to(out.device)
```

code looks like this:

```python
    def drop_mask(self, B, global_cols, n_cols):
        """ Generate drop mask; [n_cols, B].
        1) generate global masks
        2) generate local masks
        3) resurrect random path in all-dead column
        4) concat global and local masks

        Args:
            - B: batch_size
            - global_cols: global columns which to alive [GB]
            - n_cols: the number of columns of mask
        """
        # global drop mask
        GB = global_cols.shape[0]

        # calc gdrop cols / samples
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]

        # gen gdrop mask
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.

        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.-self.p_ldrop, [n_cols, LB]).astype(np.float32)
        alive_count = ldrop_mask.sum(axis=0)
        
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices] = 1.

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)
```
The parameters is:
- `B`: the batch size (!!! Note the number of block like in the paper)
- `global_cols`: dictate which column to odrop
- `n_cols`: the number of columns to mask

Remember here, we are doing per-sample drop path. Meaning that we will split the full batch of data being fed into a section that will have global drop path applied and a section with local drop path applied.

GB is the amount of the Global Drop Path Batch (which we extablished all the way back to FractalNet class)

That mask creation is then split into two section, one for global drop path and one for local drop path:
```python
# global drop mask
GB = global_cols.shape[0]

# MAKE THE GLOBAL DROP MASK

# local drop mask
LB = B - GB
# MAKE THE LOCAL DROP MASK

drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1) # CONCATENATE THE TWO!

return torch.from_numpy(drop_mask)
```
At the end the two mask are concatenated which will cover the full batch

Let's check them out one by one:
```python
        GB = global_cols.shape[0]

        # calc gdrop cols / samples
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]

        # gen gdrop mask
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.
```

First, we are subtracting from the random array the following amount: `(self.n_columns - n_cols)`.
This specific formulation is to avoid cutting off the signal of region that have nothing to do with the current join.

In our example with a singular element at the last column we would have `4-1` which gives us 3.
If the global_cols drop for a sample was 3, we would have `3-3=0`

This is then used to get a global_drop_path_indice which `gdrop_indices = np.where(gdrop_cols >= 0)[0]` which in that case it would only be true for global_cols which had the last columns index randomly sampled.

Finally, with that indice, we are able to flip the right zero into 1s to **keep alive** the column that is mentioned by global_cols.
Super important here, the index we are randomly selecting is the one that is kept alive, not the one that we cut off.

Therefore, the mask we are creating is done in the following manner:
1. put everything as dead signal (aka 0s)
2. save some signal by putting 1 instead of 0s at the right spots.

This is happening on a per sample basis throughout the whole batch, so at the end of that process we'll have a 2D array that is equal to the BATCH_SIZE-GB.

```python
        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.-self.p_ldrop, [n_cols, LB]).astype(np.float32)
        alive_count = ldrop_mask.sum(axis=0)
        
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices] = 1.

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)
```

For local drop path, we are doing it on the rest of the batch size.

in this section: `ldrop_mask = np.random.binomial(1, 1.-self.p_ldrop, [n_cols, LB]).astype(np.float32)`
We are creating a 2D grid of 1 and 0 with the probability p_ldrop.

We don't want to have 1 row that is all dead, we should at least have 1 signal that moves through.

This is what this part is doing:
```python
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices] = 1.
```
We are checking across sample if we have a spot where we simply killed off all signal across a row.
At these indices we will randomly flip 1 column back on.

Finally we concatenate both the global_drop_mask and the local_drop_mask into 1 mega mask that cover the whole batch!

Hardest part is done now, let's take a look at the convolutional block!


## ConvBlock
Here the Convolution Block is a short-hand class that has the convolution + drop out  + batch normalization and relu mix.
It's easier to package this into a singular entity since the fractal expension rule will be based off this.

```python
class ConvBlock(nn.Module):
    """ Conv - Dropout - BN - ReLU """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, dropout=None,
                 pad_type='zero', dropout_pos='CDBR'):
        """ Conv
        Args:
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal-dropout
        """
        super().__init__()
        self.dropout_pos = dropout_pos
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            # [!] the paper used reflect padding - just for data augmentation?
            self.pad = nn.ReflectionPad2d(padding)
        else:
            raise ValueError(pad_type)

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=0, bias=False)
        if dropout is not None and dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.dropout_pos == 'CDBR' and self.dropout:
            out = self.dropout(out)
        out = self.bn(out)
        out = F.relu_(out)
        if self.dropout_pos == 'CBRD' and self.dropout:
            out = self.dropout(out)

        return out
```
- We have the constructor here
- forward function.

Let's take a look at the constructor

### ConvBlock | init
```python
def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, dropout=None,
                 pad_type='zero', dropout_pos='CDBR'):
        """ Conv
        Args:
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal-dropout
        """
        super().__init__()
        self.dropout_pos = dropout_pos
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            # [!] the paper used reflect padding - just for data augmentation?
            self.pad = nn.ReflectionPad2d(padding)
        else:
            raise ValueError(pad_type)

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=0, bias=False)
        if dropout is not None and dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.bn = nn.BatchNorm2d(C_out)
```
A few argument of interesting here:
- `C_in`: channel in
- `C_out`: channel out
- `kernel_size`: size of the convolution
- `stride` : movement of the convolution
- `padding` : how much padding we should be doing
- `dropout` : percentage of drop out
- `pad_type` : either zero padding or reflection padding
- `dropout_pos` : where in the sequence conv-bn-relu are we going to put the dropout (3 choice)


in this constructor we are basically setting up these variables for the forward function:
```python
self.dropout_pos = dropout_pos
self.pad = nn.ZeroPad2d(padding) OR nn.ReflectionPad2d(padding)
self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=0, bias=False)
self.dropout = nn.Dropout2d(p=dropout, inplace=True) OR None
self.bn = nn.BatchNorm2d(C_out)
```

### ConvBlock | forward
```python
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.dropout_pos == 'CDBR' and self.dropout:
            out = self.dropout(out)
        out = self.bn(out)
        out = F.relu_(out)
        if self.dropout_pos == 'CBRD' and self.dropout:
            out = self.dropout(out)

        return out

```
The forward function is straightforward:
- we pad the input
- we pass it through the convolution layer
- we either do drop out now or after the batch normalization
- we do batch normalization then ReLU
- return the output of all this.

And voila, this is FractalNet 🎉
