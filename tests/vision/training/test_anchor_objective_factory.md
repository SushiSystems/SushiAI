# tests/vision/training/test_anchor_objective_factory.cpp


Suites: `Small_AnchorObjectiveFactory` and `Small_AnchorObjectiveFactoryOutputs`.

Tests `build_anchor_objective` and `make_anchor_objective` of `anchor_objective_factory.hpp`. The
context is a model of an 8 by 6 pixel input with a batch of 2 that publishes `head.0` (`[2, 4, 3,
14]`), `stem` (`[2, 4, 3, 8]`) and `head.1` (`[2, 2, 1, 14]`), and a task of two classes and
two heads of two anchors each. No case needs a device.

### TheDefaultAssignmentPicksTheDarknetRule

A task without an assignment member builds the best-shape rule.

### TheShapeRatioAssignmentPicksTheShapeRatioRule and TheShapeRatioAssignmentRefusesAnActivatedHead

A `shape_ratio` assignment over two heads of logits with squared sigmoid sides builds the
shape-ratio rule. With `head.1` activated it is refused, naming `head.1`.

### TheNetworkSizeIsTheInputsHeightAndWidth

The network is 8 high and 6 wide, and the layout holds the four anchors of the task.

### EachHeadTakesItsExtentsFromItsPublishedOutput

`head.0` takes a batch of 2 and a grid of 4 by 3, `head.1` a grid of 2 by 1, each by the output of
its name, though the model published `stem` between them.

### EachHeadTakesItsCodingAndScaleFromTheTask

Both heads hold two classes; `head.0` is a logit head and `head.1` an activated head with a stretch
of 1.5.

### TheSectionSetsTheTrainingOptionsOfEveryHead

A section with `box_weight` 0.5 sets it on both heads. The object weight and the box loss stay
what an empty section gives: the case builds that objective too and compares the two, so no default
is spelled in the test.

### DataThatStatesNoClassCountIsAccepted and ADataClassCountOtherThanTheTasksIsRefused

A context whose data states no class count is accepted. A count of 80 fails with `Incompatible`
naming "80 classes" and "task states 2".

### AModelWithoutATaskIsRefused

A context with no task fails with `InvalidDescription` naming `'task'` and `'anchor'`.

### AHeadTheModelDoesNotPublishIsRefused

With `head.1` removed, the failure names "head 1", `'head.1'` and what the model does publish,
`'head.0'` and `'stem'`.

### AnOutputOfOtherExtentsIsRefused

Rows: `head.0` of 15 channels, `head.1` of a batch of 3, and `head.0` of rank 2. Each fails with
`Incompatible` naming the output and `[2, H, W, 14]`; the first row also names `[2, 4, 3, 15]`.

### AnInputOfAnotherRankIsRefused

An input sample of `[192]` fails with `InvalidDescription` naming `[192]` and `[H, W, C]`.

### AKeyTheSectionDoesNotHaveIsRefused

A section with `label_smoothing` fails with `InvalidDescription` naming `objective` and the key.

### TheRegistryFormReturnsTheObjective and TheRegistryFormPrefixesARefusalWithItsContext

The registry form returns an objective named `anchor_delta`. Given a context with no task it fails
with `InvalidDescription` naming the context it was given, "the objective block", and `'task'`.
