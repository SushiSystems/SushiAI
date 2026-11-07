# tests/language/blocks/test_expert_trace.cpp


Suites: `Small_BlockExpertTrace`.

Tests the ports that `modules/language/blocks/include/SushiAI/language/blocks/expert_trace.hpp`
names. The header declares no function, so the one promise is seen through the node that
`make_moe_feed_forward_node` builds. The traces a block with experts fills are tested in
`test_moe_feed_forward.cpp` and `test_expert_transformer_block.cpp`.

## ThePortsOfABlockWithExpertsHaveDistinctNames

The node of a block with experts has three ports, and no two of them carry the same name.
