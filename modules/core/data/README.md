# data

## Owns

The training data as a model reads it. `Dataset` is the interface of a dataset: it reports its
batch size, input width, class count and batch count, and returns each `Batch` by index.
`SyntheticClassification` and `Rings` are the two synthetic datasets, built by
`SyntheticClassification::make` and `Rings::make`. `MaterialisedDataset` holds batches that are
already resident on the device. `DatasetRegistry` maps a configured type name to a factory.

## Depends on

The module declares `core` and `config` as public dependencies. It declares no private
dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/data/dataset.hpp` | `Batch`, `Dataset`, `SyntheticOptions`, `SyntheticClassification`, `RingsOptions`, `Rings`, `make_rings` |
| `SushiAI/data/dataset_registry.hpp` | `DatasetFactory`, `DatasetEntry`, `DatasetRegistry`, `register_datasets`, `make_dataset` |
| `SushiAI/data/materialised_dataset.hpp` | `MaterialisedDataset`, `allocate_batch`, `write_one_hot` |
