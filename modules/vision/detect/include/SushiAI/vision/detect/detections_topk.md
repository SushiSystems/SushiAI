# modules/vision/detect/include/SushiAI/vision/detect/detections_topk.hpp


Both postprocess paths take a whole batch: the boxes, the scores, their `PredictionShape`, then
the settings of the path. `select_top_detections` writes a fixed `[B, k, 6]` array;
`keep_confident_detections` and `select_nms_detections` write one row vector per image, because
their row counts differ between images.

## count_top_detections

Returns the rows per image, `k = min(max_det, anchors)` (`U/nn/modules/head.py:279`), and throws
on an extent or a limit that is not positive. The limit is an argument of the two top-k functions;
the reference's value is `K_MAX_DETECTIONS`.

## select_top_detections

`Detect.postprocess` on the one-to-one head, the path that is not class-agnostic and not grouped:

```
U/nn/modules/head.py:279  k = min(max_det, anchors)
U/nn/modules/head.py:285  ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1]
U/nn/modules/head.py:286  scores = self._gather(scores, ori_index)
U/nn/modules/head.py:287  scores, index = self._grouped_topk(scores.flatten(1), k, groups)
U/nn/modules/head.py:288  return scores[..., None], (index % nc)[..., None].float(), self._gather(ori_index, index // nc)
```

The k anchors of largest best-class score are kept; their k by C scores are flattened; the k
largest of those give the class as `index % C` and the anchor as entry `index / C` of the first
selection. One anchor can therefore yield several detections. The inputs are decoded pixel xyxy
boxes and probabilities, which is what `_inference` hands over (`U/nn/modules/head.py:216-217`).

## keep_confident_detections

Validation of the one-to-one head is `select_top_detections` followed by this filter at
`K_VALIDATION_CONFIDENCE`. The reference validator passes the head's rows through
`non_max_suppression` with `end2end=True` (`U/models/yolo/detect/val.py:168-176`), whose
end-to-end branch suppresses nothing and keeps the rows above the confidence:

```
U/utils/nms.py:74  mask = pred[:, 4] > conf_thres
U/utils/nms.py:77  idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
U/utils/nms.py:78  output.append(pred[idx].float())
```

A score equal to the threshold is dropped and the rows keep their order. The function takes the
rows and the batch size and nothing else: the rows per image are the row count over the batch,
and it throws when the batch does not divide them.

The `[:max_det]` cut is not ported. The caller keeps `k`, the limit it gave
`select_top_detections`, at or below the validator's row limit. Both are 300 in the reference
(`U/nn/modules/head.py:83`, `U/cfg/default.yaml:57`), so the cut removes nothing there. With a
larger top-k limit this function keeps rows the validator drops. The `classes` filter of
`U/utils/nms.py:75-76` is not ported either.

A validation consumer of the one-to-one head has to call this filter. Without it the metric is
computed over rows the reference drops: in the fixture no score reaches 0.001, so the reference
validator keeps none of the 300 rows (`validation.row_count` in `MANIFEST.json`).
