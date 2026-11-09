# tests/layers/nn/test_built_rows.cpp


Suites: `Small_LinkedRows`, `Small_LinkedRowsRefusal`, `Small_LinkedRowsTypeInUse`,
`Medium_LinkedRowsDeathTest`.

Tests `NN::Detail::read_row_above`, `NN::LinkedNodeEntry` rows of `NN::NodeRegistry` and
`NN::BuiltRows`. The file registers a linked type `follower` whose factory reads a row number
under the key `names`, fails unless that row holds an embedding and returns an `Identity`
node.

## follower_model

`follower_model` parses an embedding row, a `layernorm` row and a `follower` row whose `names` is the given row reference.

## ARowReferenceTakesThreeForms

With three rows above, `{"names": 0}` reads as row 0, `{"names": -1}` as row 2 and
`{"names": {"node": 1}}` as row 1.

## Small_LinkedRowsRefusal.ARowReferenceThatDoesNotFitIsRefused

Each reference fails with `InvalidDescription` and the row's substrings. The rows are
ARowNotAbove (row 3 of three: "layer 3", "'names'", "row 3", "not above"),
ANegativeRowBeyondTheTop (-2: "row -2"), TheRowBeforeWithNoRowAbove (-1 with no row above:
"layer 0", "0 rows are above"), AString ("row number", "string"), AnObjectWithAPort ("'port'"),
AnObjectWithNoNode ("'node'") and AnAbsentKey ("'names'", "absent").

## ALinkedTypeIsFoundAndListed

`find_linked("follower")` returns a row, `find_node("follower")` is null and `linked()` holds
that one row.

## Small_LinkedRowsTypeInUse.ATypeNameInUseIsRefused

Adding an entry under a name already in use fails with `InvalidDescription`, the quoted name and
"already". The rows are ANodeTypeAsALinkedType (a linked row named `concat`),
AnotherNodeTypeAsALinkedType (a linked row named `relu`) and ALinkedTypeAsANodeType (a
`NodeEntry` named `follower`).

## BuildModelHandsALinkedRowTheRowsAbove

`NN::build_model` over an embedding row, a `layernorm` row and a `follower` row naming row 0
returns a model of three nodes. `NN::find_embedding` returns a layer for node 0 and null for
node 1.

## BuildModelRefusesALinkedRowThatNamesANonEmbedding

The same model with `"names": -1` fails with `InvalidDescription`, "layer 2 'follower'", "row 1"
and "no embedding".

## ALinkedRowBuiltAloneSeesNoRow

`NN::make_node` for `follower` with `{"names": 0}` fails with `InvalidDescription`, "layer 0
'follower'" and "not above".

## AnUnknownTypeListsTheLinkedType

`NN::make_node` for the unknown type `missing` fails with a message containing `'missing'` and
`'follower'`.

## BuiltRowsCountsTheNodesOfItsGraph

A `BuiltRows` over a `ModelGraph` of one node has the size 1 and `node(0)` is the graph's node
0. A default `BuiltRows` has the size 0.

## ARowOutsideTheViewIsFatal

The case is in the suite `Medium_LinkedRowsDeathTest`. `node(0)` on a default `BuiltRows` stops the process with a report containing
"BuiltRows::node" and "row 0".
