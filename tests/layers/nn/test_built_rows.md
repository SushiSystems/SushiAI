# tests/layers/nn/test_built_rows.cpp


Suites: `Small_LinkedRows`, `Small_LinkedRowsDeathTest`.

Tests `NN::Detail::read_row_above`, `NN::LinkedNodeEntry` rows of `NN::NodeRegistry` and
`NN::BuiltRows`. The file registers a linked type `follower` whose factory reads a row number
under the key `names`, fails unless that row holds an embedding and returns an `Identity`
node.

## ARowReferenceTakesThreeForms

With three rows above, `{"names": 0}` reads as row 0, `{"names": -1}` as row 2 and
`{"names": {"node": 1}}` as row 1.

## ARowReferenceThatDoesNotFitIsRefused

Seven references fail with `InvalidDescription`: row 3 of three ("layer 3", "'names'", "row 3",
"not above"), -2 ("row -2"), -1 with no row above ("0 rows are above"), a string ("row number",
"string"), an object with a `port` key ("'port'"), an empty object ("'node'") and an absent key
("'names'", "absent").

## ALinkedTypeHasOneRowAmongAllKinds

`find_linked("follower")` returns a row, `find_node("follower")` is null and `linked()` holds
that one row. Adding a linked row named `concat` or `relu`, or a `NodeEntry` named `follower`,
fails with `InvalidDescription`, the name and "already".

## BuildModelHandsALinkedRowTheRowsAbove

`NN::build_model` over an embedding row, a `layernorm` row and a `follower` row naming row 0
returns a model of three nodes. `NN::find_embedding` returns a layer for node 0 and null for
node 1. The same model with `"names": -1` fails with `InvalidDescription`, "layer 2
'follower'", "row 1" and "no embedding".

## ALinkedRowBuiltAloneSeesNoRow

`NN::make_node` for `follower` with `{"names": 0}` fails with `InvalidDescription`, "layer 0
'follower'" and "not above". For the unknown type `missing` it fails with a message containing
`'missing'` and `'follower'`.

## BuiltRowsCountsTheNodesOfItsGraph

A `BuiltRows` over a `ModelGraph` of one node has the size 1 and `node(0)` is the graph's node
0. A default `BuiltRows` has the size 0.

## ARowOutsideTheViewIsFatal

`node(0)` on a default `BuiltRows` stops the process with a report containing
"BuiltRows::node" and "row 0".
