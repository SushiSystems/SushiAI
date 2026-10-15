# umbrella

## Owns

The single header that includes `<SushiAI/SushiAI.hpp>` and every public header of the language
domain. The module declares nothing and has no source of its own.

## Depends on

- `umbrella` of `layers` (`PUBLIC_DEPENDS`), which provides `<SushiAI/SushiAI.hpp>`.
- `language/text`, `language/blocks`, `language/datasets`, `language/training`,
  `language/generation` and `language/registration` (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language.hpp>` | Nothing; it includes `<SushiAI/SushiAI.hpp>` and the language headers |
