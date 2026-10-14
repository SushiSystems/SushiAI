# model_text

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The parsers that turn the JSON text a test writes into config values: `attributes_from`
for the attributes of one row and `model_from` for a whole model file. Both name the source
"<test>".

## Depends on

`config`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/model_text/model_text.hpp` | `attributes_from`, `model_from` |
