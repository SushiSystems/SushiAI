# tests/language/registration

The tests of the module `language/registration`, one file per header of `modules/language/registration/include/`. The suite prefix of a case gives its ctest label.

## test_registration.cpp

Suites: `Medium_LanguageRegistration`.

Tests the three registration entry points of the language module, `Language::register_nodes`,
`Language::register_datasets` and `Language::register_objectives`, and that model files using
their rows build.

### RegisterNodesAddsTheSixBlockRows

`register_nodes` adds five node rows in the order "position_embedding",
"causal_self_attention", "feed_forward", "transformer_block", "moe_feed_forward", and one linked
row, "language_head". A second call fails with `InvalidDescription` and a message holding
"'position_embedding'" and "already".

### RegisterDatasetsAddsTokensAndPrompt

After `register_datasets` the registry has two rows, and `find` returns a row for "tokens" and
for "prompt".

### RegisterObjectivesAddsNextTokenAndExpertBalance

After `register_objectives` the registry has two rows, and `find` returns a row for "next_token"
and for "expert_balance".

### TheNextTokenRowBuildsBesideTheRowsOfTrain

With the rows of `Train::register_objectives` and of the language module in one registry,
`Train::make_objective` builds "next_token" over an output of the shape [2, 3, 5] and five
classes, and the objective's name is "next_token". A second `Language::register_objectives`
fails with "'next_token' is already registered".

### ADecoderModelFileBuilds

A model of an embedding, a position embedding, a transformer block and a language head tied to
row 0 builds into four nodes, and `output_shape([8])` is [8, 32]. A model whose head is tied to
row 1, the position embedding, fails with `InvalidDescription`; the message holds
"layer 2 'language_head'", "'tie'", "row 1" and "no 'embedding' row".

### ADecoderWithExpertsPublishesWhatTheBalanceRowReads

A decoder with a four-expert transformer block and a two-expert `moe_feed_forward` row
publishes five outputs. Traced over a [2, 8] `Int32` input, the logits have the shape
[2, 8, 32], "balance2" has [1, 1] and "usage3" has [1, 2]. `Blocks::K_BALANCE_PORT` equals
`Training::K_BALANCE_OUTPUT_PREFIX`, `Blocks::K_USAGE_PORT` equals
`Training::K_USAGE_OUTPUT_PREFIX`, and a "sum" objective of a "next_token" term and an
"expert_balance" term builds over the published shapes.

Suites: `Medium_Moe`, in `sushiai_language_tests`.

The block cases run a FLOAT64 mixture-of-experts block of three experts with a hidden width of 3
over a `[1, 3, 4]` sequence, with every parameter filled from seeded noise. The mixture is compared
with a host computation within 1e-10. The gradient cases take the cross-entropy of the block's rows
plus 0.5 times the balance term and check it with the full-Jacobian central difference. The training
cases build a decoder of two blocks of four experts from a model file in the test, with a window of
8 and a vocabulary of 32, and fit one batch of four rows for 200 updates with AdamW at 3e-3.

### ADenseMixtureEqualsTheHostComputation

With every expert kept, the output equals the gate-weighted sum of the experts the host computes.

### AMixtureOfTheTopExpertsEqualsTheHostComputation

With one expert kept and with two, the output equals the host computation over the renormalised
gate.

### ADenseMixtureDifferentiatesAtItsRouter

The gradient of the fit and the balance at the weight of the router.

### ADenseMixtureDifferentiatesAtItsInput

The gradient of the fit and the balance at the sequence the block reads.

### AMixtureOfTheTopExpertDifferentiatesAtItsRouter

A block that keeps one expert is differentiated at the router weight. The graph holds no
MAXPOOL2D_BACKWARD node, and the gradient matches central differences.

### ADecoderOfExpertsLowersItsLossWithoutTheBalance

Trained with the `next_token` objective alone, the report holds 200 steps and one compile, and the
last loss is below half the first. The last metrics hold no `expert_usage_min`.

### TheBalanceTermKeepsEveryExpertInUse

Trained with a `sum` objective that adds the balance term, the report again holds 200 steps, one
compile and a last loss below half the first. `expert_usage_min` is above 0.10 and at most 0.25 plus
1e-5, `expert_usage_max` is at least 0.25 minus 1e-5, and the report's text names
`expert_usage_min`.

Suites: `Small_DecoderModelFile`, `Medium_DecoderFromFiles`, in
`sushiai_language_tests`.

The cases build the decoder that `decoder_model.hpp` states as a model file: a vocabulary of 32, a
window of 8, a width of 16 and two blocks, with a head tied to the token table. The graphs are
traced for a batch of 2. `Small_DecoderModelFile` needs no engine.

### TheFileStatesAnInt32Input

The spec has the input shape `[8]`, the input dtype int32 and a task that is not null.

### TheTiedHeadSharesTheTokenTable

After a probe trace the model has the parameters `0.weight`, `1.weight` and `4.norm.weight` and no
`4.linear.weight`. The parameter element count equals the count worked out by hand: `V * D + T * D +
blocks * (12 * D^2 + 13 * D) + 2 * D`. The state refuses `4.linear.weight` as `InvalidDescription`.

### TheTrainingGraphBindsEveryParameterOnce

The training graph of the next-token objective binds as many slots as the model has parameters, and
exactly one of them is named `0.weight`.
