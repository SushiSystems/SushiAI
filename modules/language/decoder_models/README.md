# language/decoder_models

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The small decoder model files the language tests build, and their extents: the vocabulary,
the window, the width and the blocks.

## Depends on

`config`, `failure`, `nn`, `language/registration` and `model_text`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/language/decoder_models/decoder_model.hpp` | The decoder model texts and their extents |
