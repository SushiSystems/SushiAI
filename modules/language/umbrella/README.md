# umbrella

## Owns

The single header that includes `<SushiAI/SushiAI.h>` and every public header of the language
domain. The module declares nothing and has no source of its own.

## Depends on

- `umbrella` of `layers` (`PUBLIC_DEPENDS`), which provides `<SushiAI/SushiAI.h>`.
- `language/text`, `language/blocks`, `language/datasets` and `language/training` (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/language.h>` | Nothing; it includes `<SushiAI/SushiAI.h>` and the language headers |
