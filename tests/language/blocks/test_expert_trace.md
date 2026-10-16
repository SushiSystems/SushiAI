# tests/language/blocks/test_expert_trace.cpp


Suites: `Small_BlockExpertTrace`.

Tests `modules/language/blocks/include/SushiAI/language/blocks/expert_trace.hpp`: the struct
`ExpertTrace` and the port names `K_BALANCE_PORT` and `K_USAGE_PORT`. The header declares no
function, so the cases pin its values. The traces a block with experts fills are tested in
`test_moe_feed_forward.cpp` and `test_expert_transformer_block.cpp`.

## ADefaultTraceHoldsNoValue

`output`, `balance` and `usage` of a default `ExpertTrace` each equal `Graph::K_NO_VALUE`.

## ThePortNamesAreDistinct

The two names are "balance" and "usage". They differ from each other and from
`NN::K_SINGLE_PORT_NAME`, the name of the first port of a block with experts.
