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

Let's look at each of the element in the reverse order to gain a detailed appreciation of the implementation.

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
Side note: great documentation here.
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

Let's explore the FractalBlock now

## FractalBlock
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

Let's look at all of them, starting with the drop mask function

## FractalBlock | join
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
- `global_cols`: ????

Here there are two main case, if we are not in training mode we will calculate the mean with no drop and generate the output for the next layer.

However, if we are in training we will follow this sequence:
1. create the mask of column to drop out using the `drop_mask` function
2. calculating how much column are still alive.
3. masking the inputs
4. calculating the means for this join operation.

It's a bit confusing here with the alive/dead nomenclature, but this part:
`n_alive[n_alive == 0.] = 1` is only to not make the division carry on zero in the denominator.

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


The whole idea of this drop_mask function is to generate a mask on the network in order to turn off some of the network columns like in the paper.

The parameters is:
- `B`: the batch size
- `global_cols`: ??????
- `n_cols`: the number of columns to mask




#

### FractalBlock | init
```python
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
```

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

## FractalNet