# tests/language/blocks/test_incremental.cpp


Suites: `Small_BlockIncremental`.

Tests the constants of `modules/language/blocks/include/SushiAI/language/blocks/incremental.hpp`.
The header names the leaves the incremental forms of the blocks are fed and the score of a masked
position. It declares no function, so the cases pin the two properties a reader of those leaves
depends on.

## EveryLeafHasItsOwnName

`K_POSITION_LEAF`, `K_ATTENTION_MASK_LEAF`, `K_CACHE_SLOT_LEAF`, `K_ROTARY_COSINE_LEAF` and
`K_ROTARY_SINE_LEAF` are five different names, none of them empty. Two leaves of one name would
make a graph feed one of them twice.

## AMaskedScoreTakesNoWeightInTheSoftmax

`K_MASKED_ATTENTION_SCORE` cast to float32 is finite and negative. Against a readable score of 0,
its softmax weight `exp(masked) / (exp(masked) + exp(0))` is exactly 0 in float32.
