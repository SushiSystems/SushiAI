# datasets

## Owns

The MNIST dataset loader. It reads the four IDX files of one split into a `Data::Dataset`, with
the tensors allocated on the `SushiBLAS::Engine` passed in. `MnistOptions` selects the directory,
the split, the flat or image layout, the batch size and the sample cap; `make_mnist` builds the
same options from a configuration section.

## Depends on

- `core`, `config` and `data` (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/datasets/mnist.hpp>` | `MnistLayout`, `MnistOptions`, `mnist_available`, `mnist_expected_paths`, `try_load_mnist`, `make_mnist` |

`try_load_mnist` returns the dataset, or a failure: `FileNotFound` when a file of the split is
absent, `InvalidFile` when one is unreadable or malformed, and `InvalidDescription` when the batch
size is not positive. `mnist_available` and `mnist_expected_paths` let a caller check the files
before it loads them.

### MnistOptions

| Field | Default | Meaning |
| --- | --- | --- |
| `directory` | `"data/mnist"` | Holds the four standard IDX files |
| `batch_size` | `64` | Samples per batch |
| `max_samples` | `4096` | Cap on the samples loaded; `0` loads all of them |
| `test_split` | `false` | Loads the test split rather than the training split |
| `layout` | `MnistLayout::Flat` | `Flat` gives `[784]` per sample; `Image` gives `[28, 28, 1]` |

### Configuration keys

`make_mnist` reads `directory` (required), `split` (`"train"` or `"test"`), `layout` (`"flat"` or
`"image"`), `batch_size` and `max_samples`. A relative `directory` resolves against the working
directory. A `split` or `layout` that names neither of its values fails with `InvalidDescription`.
`try_load_mnist` takes the engine and the options; `make_mnist` takes the engine, the attributes
and a context string.
