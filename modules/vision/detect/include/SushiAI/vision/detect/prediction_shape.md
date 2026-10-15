# modules/vision/detect/include/SushiAI/vision/detect/prediction_shape.hpp


## PredictionShape

The batch, the anchor count and the class count of `[B, A, 4]` boxes and `[B, A, C]` scores. Both
postprocess paths take one, and each throws unless its two arrays hold exactly that many values,
so a score array of another class count is refused.
