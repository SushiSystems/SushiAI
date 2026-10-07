# Closing the CV track: the layer API and the MNIST gate

**Status:** In progress. Written and syntax-checked only; the owner forbade builds and test runs on 2026-10-07 and builds himself at the end.

The IR, the gradient rules and the lowerings for convolutional networks landed in August under the
design `docs/archive/superpowers/specs/2026-08-07-cv-pipeline-layers-design.md`. No layer used
them: `nn/layers.hpp` held `Linear` and four activations, and the layer registry held `linear` and
`gelu`. This work makes a convolutional network definable from the layer API and from a JSON model
file, and trainable on MNIST through `Trainer` and `sa train`.

## 1. Gate

A CNN described by `examples/mnist_cnn/model.json` trains on MNIST through `sa train` and is
scored on the test split through `sa inference`.

The August design names a larger gate, ResNet-18 on CIFAR-10. That needs batch normalisation, a
block with a shortcut and a CIFAR-10 loader, none of which this work builds. It stays open in
`REMAINING_WORK.md`.

## 2. Decisions

The owner left these to the engineer on 2026-10-07 ("mühendis sensin SOLID'ı bozmadan
yapabilirsin"). Each is recorded with its reason so it can be reversed.

| # | Decision | Reason |
| --- | --- | --- |
| 1 | Images enter the trainer flat, `[N, features]`, and an `Unflatten` layer gives them their shape | `build_graph`, `Trainer`, `Dataset`, both applications and the model-file schema assume a rank-2 input, and several of those files call the engine and are frozen until the SushiBLAS migration. A rank-4 input leaf is the cleaner end state and is listed as remaining work |
| 2 | The runtime layer seam carries a per-sample `Shape`: `ILayer::output_shape()` is the virtual and `LayerFactory` takes the input shape | A convolution followed by a flatten cannot be chained through one `int64_t`. `output_features()` stays as a derived non-virtual, so no caller changes |
| 3 | `GraphBuilder::conv2d` owns the convolution's node sequence | CV-1 of the August design assigns it to the builder, and a second caller must not restate the weight rule |
| 4 | The convolution weight is rank 4, `[C_out, kernel_h, kernel_w, C_in]`, used through a `RESHAPE` | CV-1 says so, and the checkpoint then holds the conventional layout. CV-1 calls that reshape free; today `RESHAPE` lowers to a copy, so each step copies every kernel. Making a contiguous reshape a view is listed as remaining work in the lowering |
| 5 | One header and one source per new layer type; `layers.hpp` keeps its five types | Siblings are shaped the same from here on. Splitting `layers.hpp` is finding C12 of the estate audit and is its own change |
| 6 | Every layer type has `Shape output_shape(const Shape&) const` | Each factory asks the layer for its output shape, so no factory restates a shape rule |
| 7 | Batch normalisation is not in this work | It needs a non-trainable `Buffer`, a training and an evaluation mode, a ones initialiser and a statistics update recorded beside the graph (CV-7, CV-8). Three of those sit in frozen files |
| 8 | The `mnist` dataset row is added now, although `dataset.cpp` calls the engine | The row and its factory add no engine call, and without the row the gate cannot be reached at all |
| 9 | JSON type names follow section 11 of the August design: `conv2d`, `maxpool2d`, `avgpool2d`, `global_avgpool`, `flatten`, `upsample`, `leaky_relu` | The design committed to those names. `unflatten`, `relu`, `tanh` and `sigmoid` are added beside them |

## 3. Units

| Unit | Files |
| --- | --- |
| Shape-typed layer seam | `nn/ilayer.hpp`, `nn/layer_stack.*`, `nn/layer_registry.*`, `nn/layers.*` |
| `GraphBuilder::conv2d` | `graph/builder.hpp`, `src/graph/builder.cpp` |
| `Unflatten`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`, `Flatten`, `LeakyReLU`, `Upsample2d` | `nn/<name>.hpp`, `src/nn/<name>.cpp`, `tests/functional/unit/test_<name>.cpp` |
| Registry rows and factories | `nn/layer_registry.*`, `tests/functional/unit/test_layer_registry.cpp` |
| `mnist` dataset row | `data/dataset.hpp`, `src/data/dataset.cpp` |
| Example | `examples/mnist_cnn/` |

Every unit except the dataset row is host lane and its tests need no device.

## 4. What the owner's build must confirm

Nothing here has run. The list below is what a reading could not settle.

- The whole tree compiles; only single files were syntax-checked.
- `sa test --suite unit` for the new suites: `Unit_Unflatten`, `Unit_Conv2d`, `Unit_MaxPool2d`,
  `Unit_AvgPool2d`, `Unit_GlobalAvgPool`, `Unit_Flatten`, `Unit_LeakyReLU`, `Unit_Upsample2d`,
  `Unit_LayerRegistry`, and the changed `Unit_LayerStack`, `Unit_Module`, `Unit_Config` and
  spatial shape cases.
- A convolution step compiles and runs through a `CompiledStep`. No test before this work ran
  `IM2COL`, `MATMUL`, `BIAS_ADD` and `RESHAPE` together through the memory plan and fusion.
- `RESHAPE` requires a contiguous source. The unflatten of the input, the weight reshape and the
  flatten each depend on that holding for the value they read.
- The engine's tensor layout is row-major, which `Unflatten` and `Flatten` assume.
- The first `sa train examples/mnist_cnn/train.json` run, and its loss.

## 5. Not built

Batch normalisation with its `Buffer`, modes and update task; `Concat` (Darknet `route`) and
`BasicBlock`, which need a layer with more than one input and named children; the CIFAR-10 loader
and host-side augmentation; a rank-4 input leaf; `rematerialise_im2col` in `build_graph`; a
gradient check of a convolution's weight and bias on a device.
