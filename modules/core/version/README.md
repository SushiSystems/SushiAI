# version

## Owns

The library version, as text and as numbers. `SUSHIAI_VERSION_MAJOR`, `SUSHIAI_VERSION_MINOR`
and `SUSHIAI_VERSION_PATCH` hold the numbers, and `SUSHIAI_VERSION_STRING` holds the text.
`SUSHIAI_VERSION_ENCODE` packs three numbers into one integer, `major * 10000 + minor * 100 +
patch`, and `SUSHIAI_VERSION` applies it to the current numbers. `version_string()` returns the
text at run time.

## Depends on

The module declares no dependency, and its header includes no header of another module.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/version.hpp` | `SUSHIAI_VERSION_MAJOR`, `SUSHIAI_VERSION_MINOR`, `SUSHIAI_VERSION_PATCH`, `SUSHIAI_VERSION_STRING`, `SUSHIAI_VERSION_ENCODE`, `SUSHIAI_VERSION`, `version_string` |
