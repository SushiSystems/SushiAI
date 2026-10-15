# modules/vision/detect/include/SushiAI/vision/detect/free/level_rows.hpp


## join_level, split_level

The reference concatenates the levels on the anchor axis (`U/nn/modules/head.py:176`). SushiAI's
head outputs and loss leaves are per level, `[B, H, W, C]`. `join_level` copies one level into its
place in `[B, A, C]` for the assigner and the postprocess; `split_level` copies a slice of an
assigner output back out as one level's leaf. Both are plain copies and do not reorder inside a
level.
