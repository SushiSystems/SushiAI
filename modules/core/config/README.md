# config

## Owns

The configuration documents. `Value` is the dynamically typed value that a JSON document parses
into, and `parse_json` and `parse_json_file` build it. The model document parses into
`ModelSpec` and its `LayerSpec` rows, and the training and data documents parse into `TrainSpec`
and `DataSpec`. Each document has a version constant: `K_MODEL_VERSION`, `K_TRAIN_VERSION` or
`K_DATA_VERSION`. `find_entry` looks up a type name in a static registry table, for the modules
that select a row from a file.

## Depends on

The module declares `core` as a public dependency. Its JSON parser comes from `nlohmann_json`,
which the module takes through `EXTERNAL_PRIVATE_INCLUDES`: the module receives the include
directories and definitions of that target, and the target name stays out of the module's link
interface. It declares no other private dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/config/value.hpp` | `Kind`, `kind_name`, `Value` with its static factories `boolean`, `integer`, `real`, `string`, `array` and `object`, and its accessors `as_bool`, `as_int`, `as_real`, `as_string`, `as_array`, `has`, `at`, `at_or`, `reject_unknown_keys`, `without`, `require_version`, `parse_json`, `parse_json_file` |
| `SushiAI/config/model_config.hpp` | `K_MODEL_VERSION`, `K_MODEL_INPUT`, `SourceSpec`, `OutputSpec`, `LayerSpec`, `ModelSpec`, `parse_model`, `load_model` |
| `SushiAI/config/train_config.hpp` | `K_TRAIN_VERSION`, `K_DATA_VERSION`, `TrainSpec`, `DataSpec`, `parse_train`, `load_train`, `parse_data`, `load_data` |
| `SushiAI/config/registry.hpp` | `find_entry` |
