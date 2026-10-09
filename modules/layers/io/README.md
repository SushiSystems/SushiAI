# io

## Owns

The io module reads and writes model state on disk. `save_checkpoint` writes every entry of a
`NN::ModelState`, parameters and buffers alike, in the SACP binary format. `load_checkpoint`
reads such a file back and requires that every entry is supplied exactly once, strictly in both
directions. `checkpoint_format.hpp` defines the format, with the magic word "SACP", format
version 1, at most 4096 records per file and names of at most 256 bytes; a reader refuses any
other version. `import_weights` loads a file into a state after it rewrites each record name
through the caller's `RenameRule`s, and it fails on a record that matches no entry, repeats, or
differs in dtype or shape, and on an entry that no record supplies.

## Depends on

From `CMakeLists.txt`: `PUBLIC_DEPENDS core graph config nn`. The module's headers use two of
them:

- `core`: `Status`, `Result`, `Shape`, `DType` and `shape_of`, through `core/status.hpp`,
  `core/result.hpp`, `core/shape.hpp`, `core/common.hpp` and `core/tensor_shape.hpp`.
- `nn`: `NN::ModelState`, the state the checkpoint functions write and fill, through
  `nn/model_state.hpp`. `checkpoint.hpp` and `import_weights.hpp` include it;
  `checkpoint_format.hpp` includes only `core` headers.

No file in `io` includes a header of `graph` or `config`, so the list names two modules the code
does not use.

Two third-party libraries appear in the module without being named in the list. `checkpoint.hpp`
includes `SushiBLAS/engine.hpp`, and `checkpoint.hpp` and `import_weights.hpp` include
`SushiRuntime/core/span.hpp`. `source/import_weights.cpp` includes `SushiBLAS/tensor.hpp`. `core`
links SushiBLAS and SushiRuntime as external public dependencies, so both reach the module
through `core`.

## Public entry points

The headers sit in `modules/layers/io/include/SushiAI/io/`.

| Header | Declares |
| --- | --- |
| `checkpoint.hpp` | `save_checkpoint`, `load_checkpoint` |
| `checkpoint_format.hpp` | `K_CHECKPOINT_MAGIC`, `K_CHECKPOINT_VERSION`, `K_MAX_TENSOR_COUNT`, `K_MAX_NAME_LENGTH`, `TensorRecord`, `CheckpointWriter`, `CheckpointReader` |
| `import_weights.hpp` | `RenameRule`, `renamed`, `import_weights` |
