# hash

## Owns

The 64-bit FNV-1a hash that SushiAI's fingerprints fold through: `K_FNV1A_OFFSET_BASIS`,
`K_FNV1A_PRIME`, and the `constexpr` folds `fnv1a_byte`, `fnv1a_text` and `fnv1a_word`. A word is
folded low byte first, so a fingerprint is the same on every host. The replica kernel's model
fingerprint, the replica step's layout fingerprint, `Shape::hash`, the region signature of `eager` and
the cache file names of `vision/datasets` all use it.

## Depends on

The module declares no module dependency and links nothing. It is header-only.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/hash/fnv1a.hpp` | `K_FNV1A_OFFSET_BASIS`, `K_FNV1A_PRIME`, `fnv1a_byte`, `fnv1a_text`, `fnv1a_word` |
