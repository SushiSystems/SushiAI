# tests/vision/detect/anchor

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `anchor_delta_support.hpp` | Defines the complete IoU and its central difference the anchor delta tests expect. |
| `test_anchor_clustering.cpp` | Tests the anchors derived from seeded label boxes. |
| `test_anchor_count.cpp` | Tests the number of anchors per head chosen from seeded label boxes. |
| `test_anchor_fit.cpp` | Tests the fit report of an anchor layout on hand-built boxes. |
| `test_anchor_layout.cpp` | Tests the check of an anchor layout and the IoU of a size with no area. |
| `test_anchor_plan.cpp` | Tests the anchoring planned for a new model from seeded label boxes. |
| `test_anchor_review.cpp` | Tests the review of an anchor layout on seeded label boxes and its proposal. |
| `test_anchor_size.cpp` | Tests the IoU of two anchor sizes: hand values, order, bounds and a side not a number. |
| `test_anchor_task.cpp` | Tests that an anchor task and its task section convert both ways without loss. |
| `test_assignment_section.cpp` | Tests the "assignment" member of an anchor task section, read and written. |
| `test_assignment_thresholds.cpp` | Tests the match and ignore thresholds derived from hand-written anchor fits. |
| `test_head_assignment.cpp` | Tests the assignment of anchors to heads against the masks of five cfgs and hand cases. |
| `test_head_cost.cpp` | Tests the reported cost of an anchor head's training delta. |
| `test_head_decode.cpp` | Tests the decode of an anchor head on inputs at zero and saturation, and its row order. |
| `test_head_delta.cpp` | Tests the training delta of an anchor head: squared error, the Activated coding, the complete IoU and its gradient. |
| `test_head_options.cpp` | Tests the options of an anchor head: its channel and row counts and what it refuses. |
| `test_head_training_options.cpp` | Tests the training options of an anchor head. |
| `test_head_training_section.cpp` | Tests that training options and the anchor objective section convert both ways. |
| `test_head_truth.cpp` | Tests which truth rows an anchor head accepts and how it names the ones it refuses. |
| `test_seeded_sequence.cpp` | Tests that the seeded sequence follows SplitMix64, repeats per seed and draws once. |
| `test_shape_ratio_assignment.cpp` | Tests the cells, slots and box targets the shape-ratio assignment gives truths. |
| `test_shape_ratio_delta.cpp` | Tests the loss, the objectness targets, the factors and the delta of one anchor head under the shape-ratio assignment. |
| `test_target_assignment.cpp` | Tests the AssignmentRule interface through one rule. |
