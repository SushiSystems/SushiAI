# registration

## Owns

The two functions that fill the registries with vision rows. `register_nodes` adds the five block
node types, conv, bottleneck, c3k, c3k2 and sppf, to an `NN::NodeRegistry`. `register_datasets`
adds the mnist row to a `Data::DatasetRegistry`. Each returns a `Status`: success, or the first
failure of the registry's `add()`. The header forward-declares both registries; their
declarations live in `node_registry.hpp` and `dataset_registry.hpp`.

## Depends on

- `core`, `data` and `nn` (`PUBLIC_DEPENDS`).
- `vision/blocks` and `vision/datasets` (`PRIVATE_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/registration.hpp>` | `register_nodes`, `register_datasets` |
