# registration

## Owns

The four functions that fill the registries with vision rows. `register_nodes` adds the ten block
node types, conv, bottleneck, c3k, c3k2, c2f, c3, sppf, c2psa, detect and segment_head, to an
`NN::NodeRegistry`, in that order. `register_datasets`
adds the mnist row and the reader row to a `Data::DatasetRegistry`; the reader row's factory is
`Datasets::make_reader_dataset`, and its type is `Config::K_READER_DATA_TYPE`, which a data block
with `reader` and no `type` is given. `register_objectives` adds three rows to a
`Train::ObjectiveRegistry`, in this order: `anchor`, whose factory is
`Training::make_anchor_objective`; `pixel_cross_entropy`, whose factory is
`Training::make_pixel_cross_entropy_objective`; and `dice`, whose factory is
`Training::make_dice_objective`.
`register_importers` adds the `darknet` row, whose function is
`Importers::import_darknet_files`, to an `IO::ImporterRegistry`. Each returns a `Status`:
success, or the first failure of the registry's `add()`. The header forward-declares the four
registries; their declarations live in `node_registry.hpp`, `dataset_registry.hpp`,
`objective_registry.hpp` and `importer_registry.hpp`.

## Depends on

- `failure` (`PUBLIC_DEPENDS`).
- `config`, `data`, `nn`, `train`, `vision/blocks`, `vision/datasets`, `vision/detect` and
  `vision/training` (`PRIVATE_DEPENDS`). `config` holds `Config::K_READER_DATA_TYPE` of
  `train_config.hpp`.
- `io` and `vision/importers` (`PRIVATE_DEPENDS`), for the importer row.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/registration/registration.hpp>` | `register_nodes`, `register_datasets`, `register_objectives`, `register_importers` |
