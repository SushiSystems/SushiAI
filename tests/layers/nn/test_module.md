# tests/layers/nn/test_module.cpp


Suites: `Small_Module`.

Tests Parameter across graphs and the state walk over layers, containers and fields.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds six groups: the walk over the library's own layers and containers, the
tutorial's custom layer, the run-time arm of the walk, the per-sample output shape of `Linear`
and the four activations of `modules/layers/nn/include/SushiAI/nn/layers.hpp`, a parameter traced into more than
one graph, and the state walk over parameters, buffers and children.

## AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

## ALinearLayerNamesItsWeightAndBias

After a trace, the walk of `Linear{4, 3}` lists `weight` [3, 4] and `bias` [3], and
`parameter_elements` is 15.

## ABiasFreeLayerSimplyHasNoBiasEntry

The walk of `Linear{4, 3, false}` lists `weight` [3, 4] alone.

## SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

## NestedContainersCompoundTheirIndices

The walk of a `Sequential` holding a `Sequential` of a `Linear{4, 4}` and a `ReLU`, then a
`Linear{4, 2}`, lists `0.0.weight` [4, 4], `0.0.bias` [4], `1.weight` [2, 4] and `1.bias`
[2].

## TracingTwiceDoesNotDuplicateParameters

The second trace emits nodes but declares no new parameters, so the value count grows by
exactly the two intermediates.

## The tutorial's custom layer

`MyOffset` is the custom layer of `docs/getting_started/TUTORIAL.md` §6, compiled.

A tutorial whose code sample does not compile is worse than no tutorial, and prose cannot be
made to fail a build. So the sample lives in this file as well, and the two must be kept
identical: if you edit one, edit the other.

TheTutorialsCustomLayerTracesAndIsWalkable and TheTutorialIsRightThatMulWouldNotBroadcast also
pin the two claims §6 makes about the sample: that bias_add is required and is not a stylistic
choice, and that the two ways of getting `fields()` wrong fail differently.

## ParametersAreDeclaredAsGradientRequiringLeaves

After a trace of `Linear{4, 3}`, its weight and bias require a gradient, the weight has the
`Parameter` lifetime, and the input does not require a gradient.

## RolesAndFansAreRecordedForTheInitializer

The weight of `Linear{4, 3}` has the role `Weight`, a fan-in of 4 and a fan-out of 3, and the
bias has the role `Bias`.

## LinearRejectsAnInputWhoseInnerExtentIsWrong

`Linear{4, 3}` over a [2, 7] input throws `Error`.

## TheTutorialsCustomLayerTracesAndIsWalkable

The test checks the role the tutorial gives the parameter, because the role decides the
initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.

## TheTutorialIsRightThatMulWouldNotBroadcast

§6 says bias_add is required and not merely preferred, because `mul` goes through
`require_same` and this builder has no implicit broadcast anywhere. That is the whole reason the
sample reads the way it does, so the test asserts it and does not leave it as a remark.

## The run-time arm of the walk

`WalkableStub` exercises the third arm of the walk: a layer whose parameters are enumerated at
run time.

The stub is deliberately not a LayerStack. This file tests the walk, and the walk's contract is
`walk_parameters(prefix, ParameterVisitor)`. Anything that satisfies it is enumerated through
it, whether or not it is one of the library's own types. Testing the walk against LayerStack
would tie its test to a type that arrived two commits later.

## AWalkableLeafIsEnumeratedByItsOwnWalk

The walk of `WalkableStub` lists `weight` [2, 3] and `bias` [2], with a parameter count of 2
and 8 elements.

## AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

## TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`modules/layers/nn/include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
restates it so that the property is visible in the test that covers the walk, and so that it
names the detector and not the arm's consequences.

## Per-sample output shapes

Every layer type has `Shape output_shape(const Shape&) const`, which maps the per-sample shape
that enters the layer to the one that leaves it. A layer factory asks the layer for that shape
and does not restate the rule. The six cases below hold `Linear`, `ReLU`, `GELU`, `Tanh` and
`Sigmoid` to it, and compare it with what `forward()` traces.

## traced_shape

The file-local `traced_shape` traces a layer over a [2, features] input and returns the
result's shape.

## LinearReportsItsOutFeaturesAsThePerSampleOutputShape

`Linear{4, 3}` maps [4] to [3].

## LinearsOutputShapeAgreesWithItsTrace

The traced shape of `Linear{4, 3}` over a batch of two is [2, n], where n is the one extent
of `output_shape([4])`.

## LinearRefusesAPerSampleShapeOfTheWrongRank

`output_shape([4, 2, 1])` returns InvalidDescription with a message containing "Linear(4, 3)", "[4,
2, 1]" and "[4]". The first extent equals `in_features`, so only the rank check can refuse this
shape.

## LinearRefusesAPerSampleShapeOfTheWrongWidth

`output_shape([7])` returns InvalidDescription with a message containing "Linear(4, 3)", "[7]" and
"[4]".

## AnActivationReportsItsInputShapeAsItsOutputShape

For the shapes [5] and [4, 4, 3], each of the four activations returns the shape it was given.
The rank-3 shape is the one a convolution hands on.

## AnActivationsOutputShapeAgreesWithItsTrace

The traced shape of each activation over a [2, 5] input is [2, n], where n is the one extent
of its `output_shape([5])`.

## A parameter across graphs

The cases from AParameterTracedIntoTwoGraphsHasALeafInEach to ARoleDeclareFillsByItsRole cover
`NN::Parameter` in `modules/layers/nn/include/SushiAI/nn/module.hpp`. A parameter holds a `Graph::StateID` and
no graph. `declare()` adds a leaf to the graph of the builder it is given and binds the leaf to
the state there, so one parameter has one leaf in each graph it was traced into. `id_in(graph)`
returns that graph's leaf.

The file-local `K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is
stored and read back unchanged.

## declare_weight

`declare_weight` declares a [2, 3] parameter of role `WEIGHT` named "weight" through a given
builder and returns its leaf.

## AParameterTracedIntoTwoGraphsHasALeafInEach

The first graph holds a padding input, so the parameter's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, `state_leaf` of the
parameter's state returns the same in each graph, and the second leaf has the lifetime
`PARAMETER`.

## AParameterHasNoLeafInAGraphItWasNotTracedInto

Before any `declare()`, `id_in` is `K_NO_VALUE` and `state()` is `K_NO_STATE`. After one
`declare()` the state is set, and `id_in` of a graph that was never traced into is still
`K_NO_VALUE`.

## ASecondDeclareInOneGraphReturnsTheFirstLeaf

A second `declare()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

## ADeclareWithAnotherShapeIsRefused

After a [2, 3] declaration, a `declare()` with [4, 3] returns `InvalidDescription` with a
message containing "'weight'", "[2, 3]" and "[4, 3]", through the same builder and through a
builder over a second graph. Both graphs hold a refusal, the second holds no value and the
parameter's shape stays [2, 3].

## ADeclareWithAnotherElementTypeIsRefused

After a FLOAT32 declaration, a `declare()` with FLOAT64 returns `InvalidDescription` with a
message containing "'weight'" and both type names, in the same graph and in a second one. The
second graph holds a refusal and no value, and the parameter's type stays FLOAT32.

## ALaterDeclareKeepsTheFirstDescription

A parameter declared as a `SCALE` with a constant fill of 1.5 and fans of 4 is declared into a
second graph as a `WEIGHT` with a fan-scaled fill and fans of 9. Its fill, role and both fans
are still the first ones, and it has a leaf in the second graph.

## ACopiedLayerHasANewStateID

A copy of a traced `Linear` has a weight whose state is `K_NO_STATE` and whose `id_in(graph)`
is `K_NO_VALUE`, and `declared()` is true. Once the copy is traced, its weight has a leaf in the
graph and a state that differs from the original's, and its weight and bias leaves differ from
the original's.

## AMovedLayerKeepsItsStateID

A `Linear` move-constructed from a traced one has the same weight state and the same weight
leaf in the graph. Tracing it again adds two values, the two activations, and no parameter
leaf.

## ARoleDeclareRecordsTheFillOfItsRole

A `declare()` that states a role and no fill records a fan-scaled fill for a `WEIGHT` and a
constant fill of 0.0 for a `BIAS`.

## AnInitDeclareRecordsTheStatedFillAndTheStatedRole

A `declare()` that states both keeps both: a `BIAS` with a constant fill of 1.5 and a fan-in
of 4, and a `WEIGHT` with a fan-scaled fill. The bias's leaf requires a gradient.

## ANormalFillKeepsItsDeviation

`ParameterInit::normal(0.02)` has the kind `Normal`, the deviation 0.02 and the constant 0.
`fan_scaled()` has the kind `FanScaled` and the deviation 0. `constant_value(1.5)` has the
kind `Constant`, the constant 1.5 and the deviation 0.

## AnEmbeddingRoleWithoutAFillIsFanScaled

A parameter declared with the role `Embedding` and no fill reports that role and an init of
the kind `FanScaled`.

## ALinearDeclaresTheFillItStates

After a trace, the weight of a default `Linear{3, 4}` has an init of the kind `FanScaled`.
With `weight_init` set to `normal(0.02)` the weight's init is `Normal` with the deviation
0.02, its role is still `Weight`, and the bias's init is `Constant`.

## ARoleDeclareFillsByItsRole

The three roles give three fills: fan-scaled for `WEIGHT`, a constant 0.0 for `BIAS` and a
constant 1.0 for `SCALE`.

## The state walk

The cases from WalkStateYieldsTheThreeFieldKindsInDeclarationOrder on cover `NN::walk_state`
and the two walks derived from it. A `fields()` list holds three kinds of entry: `NN::field`
for a `Parameter`, `NN::buffer` for a `Buffer` and `NN::child` for a member that is itself
walked. `walk_state` hands each parameter and each defined buffer to an `NN::StateVisitor`.
`for_each_parameter` and `for_each_buffer` each yield one kind.

## StatePair

`StatePair` lists a parameter "scale" and a buffer "mean". `describe()` declares the parameter
as a [4] `SCALE` and defines the buffer as [4] zeros.

## MixedState

`MixedState` lists, in this order, a buffer "count", a `StatePair` child "pair", a parameter
"offset" and a buffer "unset". `describe()` defines "count" as a [1] INT64 buffer, describes
the pair and declares "offset" as a [4] `BIAS`. It leaves "unset" undefined.

## walk_all

`walk_all` collects what `walk_state` yields under a prefix as "parameter name:extents" and
"buffer name:extents" strings.

## buffer_names

`buffer_names` collects the names `NN::for_each_buffer` yields.

## WalkStateYieldsTheThreeFieldKindsInDeclarationOrder

The walk of a described `MixedState` yields "buffer count:[1]", "parameter pair.scale:[4]",
"buffer pair.mean:[4]" and "parameter offset:[4]". Parameters and buffers interleave in the
order `fields()` lists them.

## WalkStateJoinsItsPrefixToEveryName

Under the prefix "model.3" the same four entries are named "model.3.count",
"model.3.pair.scale", "model.3.pair.mean" and "model.3.offset".

## ForEachParameterNeverYieldsABuffer

`for_each_parameter` yields "pair.scale:[4]" and "offset:[4]". `parameter_count` is 2 and
`parameter_elements` is 8.

## ForEachBufferNeverYieldsAParameter

`for_each_buffer` yields "count" and "pair.mean", and `buffer_count` is 2.

## AnUndefinedBufferIsSkipped

Before `describe()` no buffer is yielded. After it two are. Once `define()` is called on
"unset", the walk yields "count", "pair.mean" and "unset".

## ADefinedBufferIsWalkedWithoutALeaf

The buffer "count" has no leaf in the graph, `id_in` is `K_NO_VALUE`, and it is one of the two
buffers counted.

## SelfWalkingStub

`SelfWalkingStub` has a member `walk_state(prefix, visitor)` and no `fields()`. It yields its
buffer "level" before its parameter "gain".

## StubPair

`StubPair` lists a `SelfWalkingStub` as the child "own" and a `WalkableStub` as the child
"legacy".

## ATypesOwnWalkStateIsTheWalk

`Detail::HasStateWalk` is true for `SelfWalkingStub` and false for `Linear`. The stub's walk
under "head" yields "buffer head.level:[3]" and then "parameter head.gain:[3]", the order its
member function states. `for_each_parameter` yields "gain:[3]" and `for_each_buffer` yields
"level".

## AChildFieldNamesEveryKindOfWalkableChild

The walk of a `StubPair` yields "buffer own.level:[3]", "parameter own.gain:[3]", "parameter
legacy.weight:[2, 3]" and "parameter legacy.bias:[2]". A `child` entry prefixes a type with its
own `walk_state` and a type with `walk_parameters` alike.

## AWalkParametersTypeYieldsNoBuffer

The walk of a `WalkableStub` yields its weight and its bias as parameters, and `buffer_count`
is 0.

## ALayerAdapterYieldsItsLayersParametersUnderThePrefix

A traced `LayerAdapter<Linear>` walked under the prefix "2" yields "parameter 2.weight:[3, 4]"
and "parameter 2.bias:[3]", and `buffer_count` is 0.

## TwoFieldsWithOneNameAreDetected

`Detail::has_unique_names` is checked in four `static_assert` lines. It is true for
`MixedState::fields()` and for an empty tuple. It is false for two `field` entries both named
"weight", and for a `buffer` and a `child` both named "pair". At run time it is true for
`Linear::fields()`. `walk_state` asserts the same predicate.

## ASequentialOfModulesKeepsItsBuilderForward

Three `static_assert` lines hold that `Sequential<Linear, ReLU>` is a `Module`, that it stays
one when nested in another `Sequential`, and that its `forward` over a `GraphBuilder` returns
a `Graph::ValueID`. A nested model of `Linear{4, 8}`, `ReLU` and `Linear{8, 3}` traces to
[2, 3] through the builder and walks as "0.0.weight:[8, 4]", "0.0.bias:[8]", "1.weight:[3, 8]"
and "1.bias:[3]".
