# modules/vision/imaging/include/SushiAI/vision/imaging/rectangle.hpp


## RectangleSettings, validate_rectangle

The reference builds its validation dataset with `rect=True` (`models/yolo/detect/train.py:76`).
The defaults are that call's: `image_size` 640, `stride` 32 (`max(model stride, 32)`,
`train.py:75`), `pad` 0.5 (`data/build.py:341`), and a batch of 32, twice the training batch
(`engine/trainer.py:317`).

## plan_rectangular_batches

`set_rectangle` (`data/base.py:405-427`). From the stored size of every image:

```
ratio[i] = height[i] / width[i]                                   (double)
order    = the indices sorted by rising ratio, equal ratios in index order
batch b  = order[b * batch_size .. (b + 1) * batch_size)
lo, hi   = the smallest and the largest ratio of batch b
unit     = (hi, 1)       if hi < 1
         = (1, 1 / lo)   else if lo > 1
         = (1, 1)        otherwise                                 (height, width)
shape[b] = ceil(unit * image_size / stride + pad) * stride         per side
```

The result is `order` and one `shape` per batch; `shape_at(position)` is the target to pass to
`ValidationPipeline::make_sample` for the image `order[position]`. With the defaults, a batch
of images 480 high and 640 wide gets a shape 512 high and 672 wide.
