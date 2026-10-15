# modules/vision/detect/include/SushiAI/vision/detect/suppression_criterion.hpp


## SuppressionCriterion, measure_suppression

`measure_suppression(box1, box2, criterion, distance_exponent)` is the number a suppression
compares with its threshold, for two x1 y1 x2 y2 boxes:

```
IoU         = intersection / union, and 0 when the intersection or the union is 0
Iou:          IoU
DistanceIoU:  IoU - (d / c)^distance_exponent, and IoU when c is 0
```

`d` is the squared distance of the two centres and `c` the squared diagonal of the smallest box
that encloses both. The IoU carries no eps, unlike `intersection_over_union` of `iou.hpp`, so
this file computes its own. `DistanceIoU` of two boxes that do not overlap is negative, so it
never exceeds a threshold of 0 or more. `K_DEFAULT_DISTANCE_EXPONENT` is 0.6.
`distance_exponent` is not checked; with an exponent of 0 the term is 1 for every pair. The
behaviour is section 4.5 of `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.
