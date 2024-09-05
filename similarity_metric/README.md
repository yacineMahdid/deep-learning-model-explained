# Intersection over Union for Box and Segmentation Masks

This code was taken mainly from the [Pytorch documentation]() and from the [instance tracker library](https://github.com/Ilyabasharov/instance_tracker/blob/92b2b5f602955df6acbc9fc282d2e4909bd47386/scripts/utils.py#L82)

## Box Intersection over Union (Box IoU)
we start out with the following function definition : `box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor`
The two tensors are of shape:
```
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
```
Where N and M are the different boxes to compare.
Here we will be comparing all `N` _boxes 1_ with all `M` _boxes 2_.
So at the end we should have:
```
Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
```

The format of any boxe is of this form:
```
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
```
Meaning that `x1`, `y1` correspond to upper left and `x2`, `y2` correspond to lower right.

This code simply implement the higher level element:
```python
    #[...]
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou
```

Therefore we need to dig into the internal function `_box_inter_union`
## _box_inter_union
```python
# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # left top
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # right bottom
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]


    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
```

First thing that this function is doing is calculating the area with the help of an helper function `box_area` for the union calculation.

## _box_inter_union : box_area
```python
[docs]def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_area)
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
```

So in a nutshell we are doing:

```python
(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
```

Which can be translated to:
```python
(x2 - x1) * (y2 - y1)
```

Coming back to the `box_inter_union`;
### box_inter_union
```python
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1) # (x2 - x1) * (y2 - y1) for all the boxes 1
    area2 = box_area(boxes2) # (x2 - x1) * (y2 - y1) for all the boxes 2

    # left top
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # right bottom
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]


    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
```

We then get all of the left top points and right bottom points for our intersection using the formula we've discussed earlier:
```python
    # left top
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # right bottom
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
```
Remember the points are arrenged (x1, y1, x2, y2)

so if we rewrite that we get:
```python
lt_x1 = max(x1_ground_truth, x1_prediction)
lt_y1 = max(y1_ground_truth, y1_prediction)

rb_x2 = min(x2_ground_truth, x2_prediction)
rb_x1 = min(y2_ground_truth, y2_prediction)
```

We then do the right bottom points minus the left top points here:
```python
wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
```
and clamp the result so that we are minimum at 0.


Removing the upcast calls and simplifying the nomenclature:
```python
wh = _upcast(rb - lt).clamp(min=0)
# similar to
wh = ((lt_x1, lt_y1) - (rb_x2, rb_y2)).clamp(min=0)
# similar to
width = (lt_x1 - rb_x2).clamp(min=0)
height = (lt_y1 - rb_y2).clamp(min=0)
```
Which is exactly the formula we've discussed in the powerpoint.

Which is then followed by the area calculation for the region that is in between
```python
inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
#similar to
inter = width*height
```
if one of these quantity is 0, then the inter_area will also be zero because there isn't an overlap.

Finally, we are calculating the union as follow:
```python
union = area1[:, None] + area2 - inter
```
Which is all the area1 (boxes1) + area2 (boxes2) and the inter region.
Remember we are dealing with batch of data here.

We finally go back to the main function:
```python
```python
    #[...]
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou
```
to do the final calculation which is inter / union and voila! 🎉

## Mask Intersection over Union (Mask IoU)
The mask IO is even simpler than the boxed version since we are dealing only with pixel and with binary masks:

```python
def mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:

    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )

    return ret
```
First thing we do here is reshape the masks so that we are dealing with vectors (i.e. the 2D information is not important)
```python
    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)
```

Then we calculate the intersection by multiplying mask1 and mask2:
```python
    intersection = torch.matmul(mask1, mask2.t())
```
Since both mask1 and mask2 are vectors now, it will return a scalar (per pair of mask)

remember it's all 0 and 1. meaning the only way the dot product will sum up a specific pixel pair is if both of them are `1`, if one of them are `0` it will sum up a `0`.

Then we calculate the area for both masks:
```python
    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)
```
We simply sum up each of the binary masks vectors into a scalar.

The union will then be the same formula as before 
```python
    union = (area1.t() + area2) - intersection
```
We sum up both areas value together and then remove the area of the intersection.

Finally we do the intersection over union:
```python
    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )
```

Here the where function work like this: `torch.where(condition, input, other, *, out=None)`

Meaning `Return a tensor of elements selected from either input or other, depending on condition`

In this case the condition is `union == 0` if this is true we will return a 0 with `torch.tensor(0., device=mask1.device)`.
Otherwise we will do the intersection over union calculation and return it `intersection / union` voila! 🎉
