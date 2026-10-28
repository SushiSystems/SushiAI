# RW-2 Make `BatchNorm2d` and a nested `ModelGraph` refuse a misfit with a `Status`

The row of this entry is in `docs/backlog/RULINGS_CARRY_OUT.md`. This file holds what did not fit the row.

- State: open
- Done when: `NN::ModuleNode::create` returns `InvalidDescription` for a `BatchNorm2d` whose input is not `[H, W, channels]` and for a nested `ModelGraph` whose input does not fit or that has no single output
- Where: `modules/layers/nn/source/batch_norm2d.cpp`, `modules/layers/nn/source/model_graph.cpp`

`ModelGraph::output_shape` refuses a graph without exactly one model output. Neither function
checks its input shape yet, so an input misfit is refused only when the model is traced.
