# tests/language/generation/test_position_feed.cpp


Suites: `Small_GenerationPositionFeedTable`, `Small_GenerationPositionFeed`, `Small_GenerationPositionFeedRefusal`.

Tests `make_position_feed` in `modules/language/generation/include/SushiAI/language/generation/position_feed.hpp` over an `InMemoryFedLeaves` and a host
engine. No step is compiled.

## AStepFedAPositionGetsTheTableFeed

The merge of two cases as rows: a step fed the position leaf gets a feed of one leaf named for it, whether the leaves also declare a ring or declare none.

## AStepFedACacheSlotGetsTheRingFeed

A step fed the slot and the angles gets a feed of three leaves. The listing of their names is in `test_ring_position_feed.cpp`.

## AStepFedBothOrNeitherIsRefused

Two rows: a step fed both leaves ("both") and a step fed neither ("neither") fail with `InvalidDescription`.
