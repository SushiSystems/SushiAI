# programs

The two drivers behind `sushiai train` and `sushiai inference`. Both read JSON configuration
and name no layer type and no dataset type.
[`applications/examples/heavy_mlp/`](../examples/heavy_mlp/README.md)
and [`applications/examples/mnist_cnn/`](../examples/mnist_cnn/README.md) each hold a
configuration to run
them with, and [`docs/guides/CLI_GUIDE.md`](../../docs/guides/CLI_GUIDE.md) documents the schema.

| File | Command |
| --- | --- |
| `train.cpp` | `sushiai train <train.json>` |
| `inference.cpp` | `sushiai inference <model.json> <model.ckpt> <eval.json>` |

The sections below hold the reasoning that the source files cite.

## train.cpp

### The property the file exists to have

The driver is written once and does not change as the library grows. It names no layer type,
no loss, no dataset and no optimizer. Each of those arrives through a registry lookup keyed
by a string out of a JSON file, so adding a layer, a detection loss or a COCO reader is a
table row and a factory function in the library, and `train.cpp` stays as it is. The
`conv2d` row of `NN::K_LAYER_REGISTRY` and the `mnist` row of `Data::K_DATASET_REGISTRY` are
rows of that kind.

The property can be checked: search the file for `Linear`, `GELU`, `Conv2d`, `CrossEntropy`,
`AdamW`, `SGD`, `Rings`, `Mnist` or `Sequential` and there is nothing to find. If an edit
adds one, the driver has stopped being what the config-driven design set out to build.

### Why it takes no flags

`applications/examples/source/demo_mlp.cpp` is the flag-driven program, and its flags are a contract the test
suite pins. In `train.cpp` every knob lives in the configuration file, because a knob that
exists in two places drifts.

### The metric belongs to the objective

`options.track_accuracy` is false. The objective owns the metric, so the trainer's own
classifier-shaped accuracy path is switched off: a driver that knows no loss type must not
compute a loss-specific number.

The epoch loop calls `run_step` itself and does not call `fit()`, so the metric it prints is
whatever the objective offered, and `std::nullopt` prints the loss alone. A detector or a
PINN reports through the same loop without the loop learning a second vocabulary.

### The compile-count line

The driver prints `compiles: N` in the shape `sa demo mlp` prints it, because the CTest case
`Integration_App.TrainRunsAndReportsOneCompile` in `tests/CMakeLists.txt` pins the line.

## inference.cpp

### Three positional arguments

The data file is one of the three positional arguments. The architecture file describes a
model and carries no data block, so without the data file there would be nothing to run on.
An optional flag that always errors when omitted would misstate what is optional.

### Forward pass only

The driver traces the forward pass and nothing else: no loss, no backward pass, no
optimizer. Inference minimises nothing, so there is no objective, and asking a configuration
for a loss the driver will not use would be asking for a value with no meaning.

Like `train.cpp`, it names no layer type and no dataset type.

### Two summaries

The driver prints the predicted-class distribution and top-1 agreement with the labels. The
distribution alone cannot tell a perfect model from a random one on class-balanced data,
because both give a uniform spread, so it is printed beside the agreement and not in its
place.

Reading `Batch::labels` names no loss and no objective. Labels are part of the `Dataset`
seam, and top-1 agreement is a property of the data and the output, not of what the model
was trained to minimise. The driver reports it only when the model's output width equals the
dataset's class count. When they differ the two are not comparable, and the driver says so
and prints no number.
