# modules/vision/imaging/include/SushiAI/vision/imaging/flip.hpp


## FlipSettings, validate_flip, plan_flip

`fliplr` is 0.5 (`cfg/default.yaml:124`). The reference decides with `random.random() < self.p`
(`data/augment.py:1588`); `plan_flip` decides with `bernoulli(probability)`, the same test. The
vertical flip is off by default (`flipud` is 0.0, `cfg/default.yaml:123`) and is not written.
The reference still draws for it; this module does not.

## apply_flip

Image: `np.fliplr` (`data/augment.py:1610`). Labels: the reference flips the centre of an xywh
box, `x_center = w - x_center`, and every vertex, `x = w - x` (`utils/instance.py:377-378`). On
corners that is `x1 = w - x2` and `x2 = w - x1`; `w` is `plan.source.width`, 640 after the warp.
