# tests/layers/train/test_sum_objective.cpp


Suites: `Unit_SumObjective`.

Tests `Train::SumObjective` and its factory `Train::make_sum_objective`, reached through the
`sum` row of the registry. Most cases sum two `CrossEntropyObjective` terms over 4 rows of 3
classes. Two local objectives stand in for the others: `StepFedObjective`, which keeps the
base's `is_fed_by_batch`, and `ReportingObjective`, whose `measure` adds the two metrics it was
made with.

## MakeKeepsItsTermsInOrder

A sum of the weights 1 and 0.5 holds two terms with those weights in order. The second term's
objective is named "cross_entropy", the sum is named "sum" and is fed by the batch, and
`leaf_prefix(1)` is "term1.".

## MakeRefusesNoTermAndAWeightItCannotUse

`make` fails with `InvalidDescription` for no term ("at least one term"), a second weight of 0
("Term 1", "'cross_entropy'", "above zero"), a first weight of -0.5 ("Term 0"), a second weight
of infinity ("Term 1", "finite") and a first weight of NaN ("Term 0").

## MakeRefusesATermItsOwnStepFeeds

`StepFedObjective` reports false for `is_fed_by_batch` and `CrossEntropyObjective` reports
true. A sum whose second term is a `StepFedObjective` fails with `InvalidDescription`, "Term
1", `'step_fed'` and "fed by its own step".

## EachTermDeclaresItsLeavesUnderItsOwnPrefix

`declare` adds two leaves, `term0.targets` and `term1.targets`, with different ids, and no leaf
named `targets`. The second has the shape [4, 3]. Two targets are required and both have the
role `K_ROLE_LABELS`.

## TheLossAddsTheWeightedLossesInTheGraph

For the weights 1 and 0.5 the loss has the shape [1] and the graph holds two NLL_LOSS nodes,
one ADD and one SCALE whose factor is 0.5. The last node of the graph is the ADD and it writes
the loss.

## ATrainingGraphDifferentiatesTheSum

`Train::build_training_graph` over one `Linear` layer and a sum of the weights 2 and 0.5 gives
a loss of one element, `spec.targets` equal to `K_NO_VALUE` and both prefixed leaves. The two
parameters each have a gradient.

## MeasureKeepsTheFirstOfTwoMetricsOfOneName

With a first term that reports `accuracy` 0.25 and `precision`, and a second that reports
`recall` 0.75 and `accuracy` 1, `measure` gives three metrics in the order `accuracy`,
`precision`, `recall`. The accuracy is 0.25 and the recall is 0.75.

## TheFactoryBuildsEachTermThroughTheRegistry

A `terms` list of two `cross_entropy` entries, the second with the weight 0.5, gives a
`SumObjective` of two terms with the weights 1 and 0.5 whose first objective is named
"cross_entropy".

## ATermMayBeASum

A `terms` list holding a `cross_entropy` entry and a `sum` entry with one `cross_entropy` term
declares two leaves, `term0.targets` and `term1.term0.targets`.

## TheFactoryRefusesASectionItCannotRead

Nine sections fail with `InvalidDescription`: no `terms` ("the objective block", "'terms'"), a
`gamma` key, `terms` as an object ("an array"), an empty list ("at least one term"), an entry
that is a number ("'terms' entry 0", "an object"), an entry without `type`, a `weight` of
"half" ("entry 0 'weight'", "a number"), the type `yolo` ("'yolo'", "'cross_entropy'",
"'sum'") and a `smoothing` key in a term ("'terms' entry 0", "'smoothing'").

## TheFactoryNeedsTheRegistryOfItsContext

`make_sum_objective` called with a context that holds no registry fails with
`InvalidDescription`, "the objective block" and "registry".
