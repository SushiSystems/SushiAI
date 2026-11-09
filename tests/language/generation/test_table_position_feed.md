# tests/language/generation/test_table_position_feed.cpp


Suites: `Small_GenerationTablePositionFeed`, `Small_GenerationTablePlace`, `Small_GenerationTableRefusal`.

Tests `TablePositionFeed` in `modules/language/generation/include/SushiAI/language/generation/table_position_feed.hpp` over an `InMemoryFedLeaves` and a host
engine. That a slide resets the rows is tested by the learned-table cases of
`test_cached_generator.cpp`.

## TheFeedListsOneInt32LeafNamedByTheStep

The feed lists one [1, 1] int32 leaf under the name of the step's leaf.

## PlaceWritesThePositionIntoItsOneLeaf

Rows: place(0) and place(5) write 0 and 5 into the leaf.

## ALeafThatDoesNotFitIsRefused

Two rows: a leaf the step is not fed and a leaf of another type each fail with `InvalidDescription`.
