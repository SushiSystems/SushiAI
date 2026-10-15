# modules/vision/detect/include/SushiAI/vision/detect/free/target_leaves.hpp


## build_target_leaves

The arrays the masked loss of SPEC section 4, "The loss as a fixed graph", reads for one branch,
over `[B, A]`. The class leaf is `Assignment::target_scores` itself and is not copied.

| Leaf | Value | From |
| --- | --- | --- |
| `weight` | sum of a row's target scores | `U/utils/loss.py:133` |
| `target_scores_sum` | `[1]`: `max(sum of all target scores, 1)` | `U/utils/loss.py:448` |
| `normalized_weight` | `weight / target_scores_sum` | `U/utils/loss.py:135` |
| `box_mask` | 1 in the four values of a foreground row, else 0 | `U/utils/loss.py:132` |
| `target_boxes` | `target_boxes / stride` on the foreground, `K_SAFE_BOX` off it | `U/utils/loss.py:461` |
| `target_distances` | `encode_distances` of that box on the foreground, 0 off it | `U/utils/loss.py:143` |
| `l1_scale` | `stride / side * 0.25 * normalized_weight`, 0 off the foreground | `U/utils/loss.py:145-154` |
| `safe_boxes` | `K_SAFE_BOX` in every row | SPEC section 8, D3 |

`side` is the image width for the left and right distances and the height for the top and bottom
(`U/utils/loss.py:146-147`). The 0.25 is the mean over four sides (`U/utils/loss.py:152`).

`K_SAFE_BOX` is `(0, 0, 1, 1)` in grid units. Ruling D3 replaces the operands of an unassigned row
with a fixed valid box before the IoU arithmetic; `safe_boxes` is that box as a full-shape leaf
for the prediction side, and `target_boxes` already carries it on the target side. Its complete
IoU with itself is finite: the union is 1 + 3e-7 and the aspect term is 0.

`target_scores_sum` is accumulated in double and narrowed once. The reference sums in float32 in
an order torch does not specify, so the two may differ in the last bits.

`l1_scale` is the SPEC's own derivation of the L1 term, not a port.

## Leaf rank and element type

The arrays are stored over the joined anchor axis, because the assigner works on it. The graph
multiplies per level, and its binary operations take operands of one shape, so each array reaches
the graph through `split_level`:

| Stored | `LevelPlacement::channels` | Leaf of one level | Arrays |
| --- | --- | --- | --- |
| `[B, A, 4]` | 4 | `[B, H, W, 4]` | `box_mask`, `target_boxes`, `target_distances`, `l1_scale`, `safe_boxes`, `AnchorLeaves::strides` |
| `[B, A, 2]` | 2 | `[B, H, W, 2]` | `AnchorLeaves::points` |
| `[B, A]` | 1 | `[B, H, W, 1]` | `weight`, `normalized_weight` |
| `[B, A, C]` | C | `[B, H, W, C]` | `Assignment::target_scores` |
| `[1]` | not cut | `[1]` | `target_scores_sum` |

`target_scores_sum` is a `std::array<float, 1>`, so an objective binds its storage as a one-element
leaf and divides the class sum by it.

Every leaf is FLOAT32. `GraphBuilder::select` takes its condition in the element type of its
branches, so `box_mask` fits a FLOAT32 loss as it is. A loss built in another element type casts
each leaf to that type with `K_CAST` before its first use; the module writes no second copy.
