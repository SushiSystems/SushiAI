# mnist_cnn: a small convolutional network on MNIST, from JSON

Three JSON files describe a two-stage convolutional classifier for the MNIST digits and the
runs that train and evaluate it. There is no C++ in this directory. `apps/train.cpp` and
`apps/inference.cpp` read the files, as they do for [`heavy_mlp`](../heavy_mlp/README.md).

No run has been made with these files. The values in them are untuned first guesses, chosen
small with the aim of a CPU run that ends in minutes, and nobody has timed one.

## The data

The loader in `src/data/dataset.cpp` reads the four IDX files under their original names,
uncompressed:

| File | Split | Holds |
| --- | --- | --- |
| `train-images-idx3-ubyte` | `train` | images |
| `train-labels-idx1-ubyte` | `train` | labels |
| `t10k-images-idx3-ubyte` | `test` | images |
| `t10k-labels-idx1-ubyte` | `test` | labels |

The repository does not ship them. Put all four in `data/mnist/` under the repository root, or
change `directory` in `train.json` and `eval.json`. When the split's two files are not in
`directory`, the driver stops with an error that names the directory, the paths it looked for
and the four file names.

`sa` starts both drivers from the repository root, so `data/mnist` is
`<repository>/data/mnist` and the paths on the command line are relative to the repository
root, wherever `sa` is called from. `src/config/train_config.cpp` resolves `model` and
`checkpoint` against the directory of the configuration file, but passes the `data` block
through as written (see `make_mnist` in `src/data/dataset.cpp`). Write an absolute `directory`
to keep the files elsewhere.

## Running it

```bash
sa build
sa train      examples/mnist_cnn/train.json
sa inference  examples/mnist_cnn/model.json \
              examples/mnist_cnn/mnist_cnn.ckpt \
              examples/mnist_cnn/eval.json
```

`sa` and `sushiai` are the same program, so both commands also run under the `sushiai` name.

The checkpoint is written beside `train.json`, because `"checkpoint": "mnist_cnn.ckpt"` is a
relative path.

## What the commands print

`sa train` prints a header with the library versions, then a line for the first epoch, every
fifth epoch and the last one, each carrying the loss and, when the objective offers one, its
metric. With 3 epochs that is epochs 1 and 3. It ends with a `compiles:` line and a `wrote`
line that names the checkpoint and its scalar count. The driver exits with an error if the
compile count is not one.

`sa inference` prints a header naming the model, the checkpoint and the data, then how many
samples and batches it evaluated, then a table with one row per output carrying the count of
samples predicted as that output and their share of the total. The model has ten outputs and
MNIST has ten classes, so the driver also prints top-1 agreement with the labels. It ends with
its own `compiles:` line.

This file quotes no loss, accuracy or timing, because none has been measured.

## The three files

| File | Describes |
| --- | --- |
| `model.json` | the architecture |
| `train.json` | the objective, the data, the optimizer, the epochs, the checkpoint |
| `eval.json` | the data to evaluate on |

`model.json` builds this chain. Activations are NHWC, and the shapes below leave out the batch
axis.

| Layer | Output shape |
| --- | --- |
| input | `[784]` |
| `unflatten` to 28 x 28 x 1 | `[28, 28, 1]` |
| `conv2d`, 8 filters, kernel 3, padding 1 | `[28, 28, 8]` |
| `relu` | `[28, 28, 8]` |
| `maxpool2d`, kernel 2, stride 2 | `[14, 14, 8]` |
| `conv2d`, 16 filters, kernel 3, padding 1 | `[14, 14, 16]` |
| `relu` | `[14, 14, 16]` |
| `maxpool2d`, kernel 2, stride 2 | `[7, 7, 16]` |
| `flatten` | `[784]` |
| `linear`, 10 units | `[10]` |

`train.json` loads the first 8192 training images in batches of 64 and runs 3 epochs of AdamW
with cross-entropy. `eval.json` loads the first 2048 test images. `max_samples` sets both caps;
the loader takes the first images in file order and drops a final partial batch.

## Limits

- Images enter flat. The loader hands each image over as 784 values, `input.features` is 784,
  and `unflatten` must be the first layer so the convolutions see `[28, 28, 1]`. A `conv2d`
  placed first receives a feature vector, not an image.
- There is no batch normalisation layer yet, so the model has none.
- The loader scales pixels to the range 0 to 1 and does nothing else: no mean and deviation
  normalisation, no shuffling, no augmentation.
- An unknown key in any block is an error.
  [`docs/guides/CLI_GUIDE.md`](../../docs/guides/CLI_GUIDE.md) documents the schema.
