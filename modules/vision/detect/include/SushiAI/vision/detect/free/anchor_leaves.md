# modules/vision/detect/include/SushiAI/vision/detect/free/anchor_leaves.hpp


## build_anchor_leaves

The loss graph decodes a level's `[B, H, W, 4]` distances into corners with `K_SUB` and `K_ADD`, and
both take operands of one shape. The anchor points therefore reach the graph with a batch axis.
`build_anchor_leaves` writes them once per image:

| Leaf | Joined | Per level | Value |
| --- | --- | --- | --- |
| `points` | `[B, A, 2]` | `[B, H, W, 2]` | cell centre, x then y, grid units, as `make_anchors` writes it |
| `strides` | `[B, A, 4]` | `[B, H, W, 4]` | the anchor's stride in each of the four values |

`split_level` with 2 and 4 channels cuts them into a level's leaves, the same way it cuts the
target leaves. `points` is what the two halves of the distances are subtracted from and added to
(`U/utils/tal.py:470-471`). `strides` has the shape of the decoded box it multiplies
(`U/nn/modules/head.py:226`); inside one level it is a single value, so a per-level graph can use
`K_SCALE` by the level's stride and leave this leaf unbound.

Both arrays depend on the input size and the batch size alone. A trainer builds them once and
rebuilds them when either changes. The call throws on a batch below 1 and on an empty level list.
