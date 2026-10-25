# modules/vision/detect/include/SushiAI/vision/detect/top_k.hpp


## select_top_k

Returns the indices of the k largest values, largest first, and orders equal values by ascending
index. A NaN ranks below every number, so the order is total whatever the data holds.
`torch.topk` and `argsort` leave the order of equal values unspecified, so this rule is
SushiAI's own, and it is where a result can differ from the reference. Four call sites inherit it:

| Call site | Reference line | What a tie decides |
| --- | --- | --- |
| first top-k of the assigner (`assigner.cpp`, `select_candidates`) | `U/utils/tal.py:263` | which candidates of equal metric a box keeps |
| second top-k of the assigner (`assigner.cpp`, `keep_second_top_k`) | `U/utils/tal.py:376` | which held anchors of equal metric survive `topk2` |
| candidate limit of the suppression (`detections_nms.cpp`, `select_image_detections`) | `U/utils/nms.py:141` | which boxes of equal score pass `max_candidates` |
| visiting order of the suppression (`detections_nms.cpp`, `suppress_overlaps`) | `U/utils/nms.py:269` | which of two overlapping boxes of equal score is kept |

The two selections of `select_top_detections` (`U/nn/modules/head.py:285-287`) use it as well.

It matters in the fixture. The reference model has random weights and nearly constant scores: 255
of the 300 rows of `one2one.detections` share their score with another row, and 11 anchor and
class pairs tie for the last 5 places. `test_detect_postprocess.cpp` therefore compares the score
and class columns row by row, compares the rows above the last score as a set, and checks that
every row is some anchor's box with that anchor's score.
