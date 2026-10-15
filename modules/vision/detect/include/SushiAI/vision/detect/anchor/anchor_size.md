# modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_size.hpp


## AnchorSize, shape_iou

`AnchorSize` is a width and a height as fractions of the network input. A label box takes the
same form. `shape_iou(a, b)` is the IoU of two sizes laid on one centre:

```
intersection = min(a.width, b.width) * min(a.height, b.height)
union        = a.width * a.height + b.width * b.height - intersection
shape_iou    = intersection / union, and 0 when union is not above 0
```

It is float32 and carries no eps. Two sizes without area give 0, where the quotient would be 0/0.
