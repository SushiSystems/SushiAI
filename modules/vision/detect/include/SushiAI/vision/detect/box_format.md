# modules/vision/detect/include/SushiAI/vision/detect/box_format.hpp


## xyxy_to_xywh, xywh_to_xyxy

```
U/utils/ops.py:257  y[..., 0] = (x1 + x2) / 2  # x center
U/utils/ops.py:259  y[..., 2] = x2 - x1  # width
U/utils/ops.py:281  wh = x[..., 2:] / 2  # half width-height
U/utils/ops.py:282  y[..., :2] = xy - wh  # top left xy
U/utils/ops.py:283  y[..., 2:] = xy + wh  # bottom right xy
```
