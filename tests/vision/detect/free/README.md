# tests/vision/detect/free

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `compile_assignment.cpp` | Checks at compile time that an assignment moves, as the assigner hands it to its caller. |
| `test_anchor_leaves.cpp` | Tests the anchor leaves of a one-image batch and their rebuild for a smaller batch. |
| `test_anchors.cpp` | Tests the anchor grid and its leaves against hand cases. |
| `test_assigner.cpp` | Tests label assignment and the loss leaves against hand cases and the reference. |
| `test_assigner_config.cpp` | Tests that both assigner presets carry the class count and refuse a count below one. |
| `test_box_coding.cpp` | Tests box coding against hand cases and the reference decode. |
| `test_distribution_expectation.cpp` | Tests the expected distances of bin logits against hand-worked distributions. |
| `test_distribution_targets.cpp` | Tests the bin weights of the distribution focal loss against hand-worked targets. |
| `test_free_task.cpp` | Tests that an anchor-free task and its task section convert both ways without loss. |
| `test_head_contract.cpp` | Tests the port names and the class bias an anchor-free head and its training share. |
| `test_level_rows.cpp` | Tests that joining a level leaves other rows alone and that splitting undoes a join. |
| `test_target_leaves.cpp` | Tests the leaf constants, the extents of every leaf and the refused geometry. |
