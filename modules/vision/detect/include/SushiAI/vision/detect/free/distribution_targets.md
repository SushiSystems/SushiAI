# modules/vision/detect/include/SushiAI/vision/detect/free/distribution_targets.hpp


## fill_distribution_weights

The leaf the distribution focal loss multiplies `log_softmax` by, so the loss is
`-sum(log_softmax(bins) * weights)` with no gather. Per side, after Ultralytics' `DFLoss` and
`BboxLoss`:

| Step | Value |
| --- | --- |
| clamp | `t` into `[0, reg_max - 1 - 0.01]` |
| bins | `tl = floor(t)`, `tr = tl + 1` |
| weights | `wl = tr - t` at `tl`, `wr = 1 - wl` at `tr` |
| scale | both times `normalized_weight / 4`: the mean over sides, the target score weight, and the division by `target_scores_sum` |

A background row has `normalized_weight` 0, so all its bins are 0. The gain is the caller's.
