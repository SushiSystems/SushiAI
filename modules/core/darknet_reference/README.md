# darknet_reference

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The readers of the files a Darknet run dumped outside the repository: the tensors, the maps
and the settings of a run, found under the folder `SA_REFERENCE_DATA_DIR` names. A test whose
file is absent reports itself skipped. Darknet is an oracle run outside the repository; nothing
of its source enters it.

## Depends on

`config`, for the settings read as `Config::Value`, `core`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/darknet_reference/darknet_reference.hpp` | The dump readers and their file lookup; see the header |
