# tests/layers/train/test_objective_output.cpp

Suites: `Small_ObjectiveOutput`, `Small_ObjectiveOutputStep`, `Small_RowMaskKey`, `Small_RowMask`,
`Small_ObjectiveOutputKinds`, `Small_ObjectiveTarget`.

Tests `Train::ObjectiveOutput`, `Train::ObjectiveTarget`, `Train::read_selection` and
`Train::RowMask`, and what the four objectives `cross_entropy`, `binary_cross_entropy`, `multi_label`
and `mean_squared_error` do with the keys `output`, `type_mask` and `target`. No case needs a device. The loss cases run the loss graph on
`Graph::HostEvaluator`; the feed cases allocate their tensors on the engine of a
`HostExecutionBackend`.

`Small_ObjectiveOutputKinds` is a parameterised suite with one row per objective: its factory, the
columns of its output (3, 1, 3 and 2), its classes, a [4, width] targets leaf and the rounded
operations of one loss term. Every row reads a model of two outputs, `first` and `second`, whose
values come from a fixed pattern of 11 values. The closed form of a row is the cross entropy
`-sum t (x - log sum exp x)` for `cross_entropy`, the mean of `max(x, 0) - x t + log(1 + exp(-|x|))`
over its elements for the two logistic objectives, and the mean of `(x - t)^2` for the regression.
The loss of a mask is the mean of the closed forms of the kept rows. The bound of a loss
comparison is twice the operations of its longest path, one for the closed form and one for the
graph: one term, the product with its weight, the sum of the 4 * width elements that rounds one
time fewer than their count, and the scale.

## TheDefaultSelectorReadsTheOnlyOutput

The default selector is not named and returns the value of a model that publishes one output. For
a model that publishes `first` and `second` it fails with `InvalidDescription`, "exactly one model
output" and both names.

## ANamedSelectorReadsItsPortAmongSeveral

The selector named `second` returns the value published under that name. The one named `third`
fails with `InvalidDescription`, "no output named 'third'" and both published names.

## ANamedSelectorReadsItsTensorAmongTheOutputsOfAStep

Over a step with the outputs `first` and `second`, the selector named `second` returns the address of
the second tensor. The selector named `third` fails with "2 outputs" and "'third'"; the default
selector fails with "one model output" and 2.

## AnEmptyNameAndANumberAreRefused

`ObjectiveOutput::named` refuses "" with "at least 1 character". `ObjectiveOutput::read` refuses
`{"output": 3}`, naming "objective" and "'output'", and `{"output": ""}`.

## TheSelectionKeysAreReadAndAStrayKeyIsRefused

`read_selection` over `{"output": "second", "type_mask": true}` gives the selector named `second`
and an active mask. Over `{"threshold": 0.9}` with the own key `threshold` it gives the default
selector and an off mask. `is_selection_key` is true for `output` and `type_mask` and false for
`threshold`. The key `huber` is refused with "'huber'", "'output'" and "'type_mask'", and with
"'threshold'" as well when `threshold` is the own key.

## TheKeyReadsOffOneValueAndColumns

The key `type_mask` absent or false gives an off mask. `true` gives an active mask of 0 columns
and column 0, a role of one value per sample. `{"columns": 4, "column": 2}` gives 4 columns and
column 2; `{"columns": 4}` gives column 0.

## TheKeyRefusesWhatNoMaskCanBe

Each failure is `InvalidDescription`. The string "yes" is refused with "objective", "'type_mask'"
and "boolean". An object without `columns` names "'columns'". `{"columns": 2, "column": 2}` is
"no column 2" and `{"columns": 0}` is "at least 1". The object key `rows` is refused. `RowMask::make(-1, 0)`
fails with "got -1".

## TheTargetKeyNamesTheRoleAnObjectiveReads

`read_selection` over `{"output": "second", "target": "choice"}` gives a named target whose
`role_or("labels")` is `choice`; over `{}` it gives the default target, whose `role_or("labels")` is
`labels`. `ObjectiveTarget::named("number")` reads `number`, and `is_selection_key` is true for `target`.

## ATargetThatIsEmptyOrNoStringIsRefused

`ObjectiveTarget::named("")` and a `target` of `""` fail with `InvalidDescription` and "at least 1
character". A `target` of 3 fails naming "objective", "'target'" and "batch role". The stray key
`huber` is refused with a message that lists `'target'` among the keys read.

## TheFactoryRefusesAMissingOutputAndAnUnnamedChoice

For each of the four objectives, the key `output` set to `third` over a model of two outputs fails
with "objective", "'third'" and "2 outputs". The three element objectives also refuse a model of
two outputs with no `output` key: "publishes 2" and "'output'". The cross-entropy factory reads no
output shape of its own when none is named, so its refusal comes from its loss; the default
selector case above covers it.

## TheNamedPortGivesTheLossOfAModelPublishingOnlyIt

The loss of the objective with `output` set to `second`, over a model of two outputs, equals the
loss of the default objective over a model that publishes only the `second` values. The equality
is exact: both graphs run the same operations on the same numbers. The loss of the objective that
names `first` differs from it, so the selector, not the data, makes the equality.

## WithoutTheKeysNoMaskLeafOrRoleIsDeclared

The objective with no key declares no `type_mask_weights` leaf. With `type_mask: true` the leaf
exists with the shape [4, width], and the required roles are those of the objective plus
one: the role `type_mask`, an empty sample shape, float32.

## AMaskOfColumnsRequiresARoleOfThatSampleShape

With `{"columns": 5, "column": 3}` the added role has the sample shape [5].

## AMaskKeepsTheMeanOverTheKeptRows

With the mask 1, 0, 1, 1 and `output` set to `second`, the loss equals the mean over rows 0, 2 and
3 of the closed form of a row, within the bound above. The weights written to the leaf are 1/3 for a
kept row and 0 for the dropped one, in double.

## AMaskThatKeepsEveryRowGivesTheUnmaskedLoss

With the mask 1, 1, 1, 1 the loss equals the loss of the objective that has no mask, within the same
bound. The two graphs add the same terms; the masked one multiplies each by 1/4.

## AMaskThatDropsEveryRowGivesLossZeroForAnyOutput

With the mask 0, 0, 0, 0 the loss is 0 for the port values of the pattern, for the same values
plus 0.5 and for those times 40. A loss that is 0 for every value of its output has a gradient of 0
with respect to it. The host evaluator has no backward rules, so the gradient graph is not run;
that claim rests on the loss being a sum of terms each multiplied by an exact 0.

## ADroppedRowDoesNotMoveTheLossAndAKeptRowDoes

With the mask 1, 0, 1, 1, adding 3 (column + 1) to row 1 leaves the loss bit for bit as it was.
Adding the same to row 0 changes it. The offset is not one value per row, because the softmax of
`cross_entropy` ignores a shift that is the same in every column.

## WithoutATargetTheObjectiveRequiresItsOwnRole

`Small_ObjectiveTarget` is one row for each of the four objectives, over the same data as the kinds
suite. With no `target` the objective requires one role: `labels`, or `values` for the regression.

## ANamedTargetIsTheRoleTheObjectiveRequires

With `{"target": "mine"}` the one role an objective requires is `mine`, for each of the four.

## FeedReadsTheNamedRoleAndNotTheDefaultOne

With `{"target": "mine"}` the objective fed a batch that holds only `mine` writes the targets leaf
with the row's targets: the one-hot rows of the classes that the batch holds as int32 for the cross
entropy, the 0 and 1 of the binary labels, the multi-label rows and the regression values, over a leaf
that held 7s. Fed a batch that holds only the default role, it fails with `InvalidDescription` and
"'mine'".

## AKeptRowGetsOneOverTheKeptCountAndADroppedRowZero

`RowMask::feed` over the mask 1, 0, 1, 1 writes into a [4, 3] leaf the float32 nearest 1/3 in the
12 elements of rows 0, 2 and 3 and 0 in the 3 elements of row 1.

## AMaskOfColumnsReadsItsColumn

A role of the sample shape [2] with the rows (1, 0) and (0, 1), read at column 1, keeps row 1
alone: the leaf holds three 0s and three 1s.

## ABatchThatDropsEveryRowWritesZeroWeights

A mask of four 0s writes 12 values of 0 to the leaf, and no failure.

## AMaskValueThatIsNotZeroOrOneIsRefused

A mask of 1, 0.5, 1, 0 fails with `Incompatible`, "row 1" and "0 or 1"; the leaf still holds twelve 7s.

## ABatchWithoutTheRoleOrWithAnotherRowCountIsRefused

A batch that carries only the labels role fails with `InvalidDescription` naming "'type_mask'". A
mask of 5 values for a leaf of 12 elements fails with `Incompatible`, "5 rows" and "holds 12". The
leaf still holds twelve 7s after both.

## AnObjectiveFeedWritesTheTargetsAndTheWeights

A `BinaryCrossEntropyObjective` with a mask, fed a batch of the labels 1, 0, 0, 1 and the mask
0, 1, 1, 0, writes the labels to `targets` and the weights 0, 0.5, 0.5, 0 to the weights leaf. The
weights are exact in float32.
