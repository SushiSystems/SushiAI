# tests/layers/train/test_sum_objective.cpp


Suites: `Small_SumObjective`, `Small_SumObjectiveWeight` (one row per refused weight pair),
`Small_SumObjectiveGradient` (one row per parameter), `Small_SumObjectiveSection` (one row per
refused section) and `Medium_SumObjectiveDeathTest`.

Tests `Train::SumObjective` and its factory `Train::make_sum_objective`, reached through the
`sum` row of the registry. Most cases sum two `CrossEntropyObjective` terms over 4 rows of 3
classes. Every other term is a scripted `Train::InMemoryObjective`: the term that its own step
feeds keeps the default `is_fed_by_batch` of false, and the terms that state and report metrics
are scripted to be fed by a batch, to state names and to append fixed values. The file holds no
`Objective` subclass.

## K_WEIGHTED_LOSS_OPERATIONS

22 rounded operations of double on the longest path to the weighted loss of 4 rows of 3 classes:
the log-softmax rounds 7 (the shift by the largest logit, the exponential, two additions, the
logarithm, the sum with the largest logit, the difference), the product with the target rounds
once, the sum of the 12 products 11 times and the mean once, which is 20 for a term. The scale by
0.5 is exact, the sum of the two terms rounds once and the decimal literal of the expected value
once.

## MakeKeepsItsTermsInOrder

A sum of the weights 1 and 0.5 holds two terms with those weights in order. The second term's
objective is named "cross_entropy".

## ASumIsNamedSum, ASumIsFedByABatch and TheLeafPrefixOfATermIsItsIndex

The sum is named "sum", is fed by the batch, and `leaf_prefix(1)` is "term1.".

## MakeRefusesNoTerm

`make` of no term fails with `InvalidDescription` and "at least one term".

## MakeRefusesAWeightItCannotUse

One row per weight pair: a first weight of -0.5 ("Term 0", "at least zero"), a second weight of
infinity ("Term 1", "finite") and a first weight of NaN ("Term 0", "finite"). A weight of 0 is not
a row any more: see `MakeAcceptsAWeightOfZero` (RW-84).

## MakeAcceptsAWeightOfZero

A sum of the weights 1 and 0 is made, and its second term holds the weight 0.

## MakeRefusesASumWhoseWeightsAreAllZero

A sum of the weights 0 and 0 fails with `InvalidDescription`, "Every term", "weight 0" and "no
parameter has a gradient": its loss would be 0 for every input.

## ATermOfWeightZeroAddsNothingToTheLoss

On the data of `TheLossAddsTheWeightedLossesOfItsTerms` with the weights 1 and 0, the loss is that
of term 0, `ln 5 - ln 3 = ln (5 / 3)`, within the rounded operations of the weighted case.

## MakeRefusesATermItsOwnStepFeeds

A sum whose second term is an `InMemoryObjective` fails with `InvalidDescription`, "Term 1",
`'in-memory'` and "fed by its own step".

## EachTermDeclaresItsLeavesUnderItsOwnPrefix

`declare` adds two leaves, `term0.targets` and `term1.targets`, with different ids. The second
has the shape [4, 3].

## NoLeafIsDeclaredWithoutAPrefix

No leaf is named `targets`.

## EachTermRequiresItsRole

Two targets are required and both have the role `K_ROLE_LABELS`.

## TheLossAddsTheWeightedLossesOfItsTerms

The loss of the weights 1 and 0.5 runs on `Graph::HostEvaluator`, so the case is `Small_`. Every
row of the logits is {0, ln 3, 0}, so its log-probabilities are -ln 5, ln 3 - ln 5 and -ln 5.
Term 0 targets class 1 in every row, and its loss is ln 5 - ln 3. Term 1 targets class 0, and its
loss is ln 5. The sum is ln 5 - ln 3 + 0.5 ln 5 = 1.5 ln 5 - ln 3 = 1.3155445799830407. The old
case counted nodes by operation and read the last node's wiring, which a graph pass would
change with the value intact.

## ATrainingGraphOfTheSumHasAScalarLoss, ...HasNoTargetsValue and ...HoldsTheLeavesOfEachTerm

`Train::build_training_graph` over one `Linear` layer and a sum of the weights 2 and 0.5 gives a
loss of one element, `spec.targets` equal to `K_NO_VALUE`, and both prefixed leaves.

## ATrainingGraphDifferentiatesTheSum

One row per parameter, 0 and 1: its gradient is set.

## MeasureReportsTheAccuracyOfTheFirstOfTwoTerms

`SumObjective::measure` keeps the first term's metric when two terms report one name (the rule is
written at `sum_objective.cpp`, in `measure`). Two scripted `InMemoryObjective` terms that state
no name and append `accuracy` report 0.25 and 0.75. The sum gives one metric, `accuracy`, of
0.25, so the case shows which term's value won.

## ATermWhoseMetricsAreAppendedAlreadyIsNotMeasured

Two scripted `InMemoryObjective` terms, each stating and appending `mean_iou`, with the values
0.25 and 0.75 (RW-76). After `measure` the first term's `measure_count()` is 1 and the second's is
0, and the sum holds one metric of 0.25.

## ATermWithANameNotAppendedYetIsMeasured

The first term states and appends `accuracy`; the second states and appends `accuracy` and
`recall`. The second is measured once, `accuracy` keeps the first term's 0.25 and `recall` is the
second's 0.75.

## ATermThatStatesNoNameIsMeasured

The second term states no name and appends `accuracy`. It is measured once, and `accuracy` keeps
the first term's value: a term that states nothing cannot be left out.

## ASumStatesTheNamesOfItsTermsOnce

Terms that state `accuracy` and `accuracy`, `recall`: the sum states `accuracy`, `recall`, so a sum
nested in a sum is left out the same way.

## ASumWithATermThatStatesNoNameStatesNone

A term that states nothing makes the sum state nothing, since the sum cannot say what it appends.

## ANullTermStopsTheProcessNamingIt

`SumObjective::make` stops the process on a null term, and the message names its index ("Term 1")
and says it is null. A death test, so it is `Medium_`.

## TheFactoryBuildsEachTermThroughTheRegistry

A `terms` list of two `cross_entropy` entries, the second with the weight 0.5, gives a
`SumObjective` of two terms with the weights 1 and 0.5 whose first objective is named
"cross_entropy".

## ATermMayBeASum

A `terms` list holding a `cross_entropy` entry and a `sum` entry with one `cross_entropy` term
declares two leaves, `term0.targets` and `term1.term0.targets`.

## TheFactoryRefusesASectionItCannotRead

One row per section, each failing with `InvalidDescription`: no `terms` ("the objective block",
"'terms'"), a `gamma` key ("'gamma'", "'terms'"), `terms` as an object ("an array"), an empty
list ("at least one term"), an entry that is a number ("'terms' entry 0", "an object"), an entry
without `type` ("'type'"), a `weight` of "half" ("entry 0 'weight'", "a number"), the type `yolo`
("'yolo'", "'cross_entropy'", "'sum'") and a `smoothing` key in a term ("'terms' entry 0",
"'smoothing'").

## TheFactoryNeedsTheRegistryOfItsContext

`make_sum_objective` called with a context that holds no registry fails with
`InvalidDescription`, "the objective block" and "registry".
