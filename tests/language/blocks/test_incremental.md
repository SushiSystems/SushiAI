# tests/language/blocks/test_incremental.cpp


Suites: `Small_BlockIncremental`.

Tests the leaf names of `modules/language/blocks/include/SushiAI/language/blocks/incremental.hpp`.
The header names the leaves the incremental forms of the blocks are fed. It declares no
function, so the one case pins the property a reader of those leaves depends on. The masked score
is tested through the generator, in `tests/language/generation/test_cached_generator.cpp`.

## EveryLeafHasItsOwnName

`K_POSITION_LEAF`, `K_ATTENTION_MASK_LEAF`, `K_CACHE_SLOT_LEAF`, `K_ROTARY_COSINE_LEAF` and
`K_ROTARY_SINE_LEAF` are five different names, none of them empty. Two leaves of one name would
make a graph feed one of them twice.
