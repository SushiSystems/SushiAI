# modules/vision/detect/include/SushiAI/vision/detect/free/assignment.hpp


## Assignment

The five outputs of `TaskAlignedAssigner.forward` at full shape over all anchors, so a fixed-shape
loss can feed them as leaves. `target_boxes` are in pixels. A row off the foreground holds the box
and the label of ground-truth row 0 of its image and index 0, as in the reference
(`U/utils/tal.py:297-300`, `381`).
