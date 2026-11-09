# tests/language/registration/test_registration.cpp


Suites: `Small_LanguageRegistration`, `Small_DecoderFromFiles`, `Medium_DecoderFromFiles`,
`Medium_Moe`, `Medium_MoeMixture`, `Medium_MoeRouterGradient`, in `sushiai_language_tests`.

Tests the three registration entry points of the language module, `Language::register_nodes`,
`Language::register_datasets` and `Language::register_objectives`, that model files using their
rows build, and the mixture-of-experts block the rows build. The `Small_` cases need no engine.
The `Medium_` cases run a kernel against a host computation or train on the device.

## Small_LanguageRegistration

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
classes, and the objective's name is "next_token".

### TheNextTokenRowIsRefusedASecondTime

With the same two registrations, a second `Language::register_objectives` fails with
`InvalidDescription` and a message holding "'next_token' is already registered".

### ADecoderModelFileBuilds

A model of an embedding, a position embedding, a transformer block and a language head tied to
row 0 builds into four nodes, and `output_shape([8])` is [8, 32].

### AnUntiedHeadIsRefusedNamingTheLayerAndTheRow

A model whose head is tied to row 1, the position embedding, fails with `InvalidDescription`; the
message holds "layer 2 'language_head'", "'tie'", "row 1" and "no 'embedding' row".

### ADecoderWithExpertsPublishesWhatTheBalanceRowReads

A decoder with a four-expert transformer block and a two-expert `moe_feed_forward` row
publishes five outputs. Traced over a [2, 8] `Int32` input, the logits have the shape
[2, 8, 32], "balance2" has [1, 1] and "usage3" has [1, 2]. A "sum" objective of a "next_token"
term and an "expert_balance" term builds over the published shapes, so the objective finds the
outputs the block publishes. The port names against the prefixes are not compared here; the
fit with the balance term below depends on them.

## Medium_MoeMixture and Medium_MoeRouterGradient

The block cases run a FLOAT64 mixture-of-experts block of three experts with a hidden width of 3
over a `[1, 3, 4]` sequence, with every parameter filled from seeded noise. The mixture is compared
with a host computation by `TestSupport::unit_roundoff`: 1048576 times the double unit roundoff.
The gradient cases take the cross-entropy of the block's rows plus 0.5 times the balance term and
check it with the full-Jacobian central difference.

### AMixtureEqualsTheHostComputation

Rows over the experts kept: every expert (`top_k` 0, seed 9100), the top 1 (seed 9200) and the top 2
(seed 9300). The output, the usage of each expert and the balance equal the host computation, which
sums the experts one at a time over the renormalised gate.

### AMixtureDifferentiatesAtItsRouter

Rows over the experts kept: every expert (seed 9400) and the top 1 (seed 9600). The gradient of the
fit and the balance at the weight of the router matches the central differences. With one expert
kept the mask is constant around the point.

## Medium_Moe

### ADenseMixtureDifferentiatesAtItsInput

The gradient of the fit and the balance at the sequence the block reads, with every expert kept.

### ADecoderOfExpertsLowersItsLossWithoutTheBalance

The cases below build a decoder of two blocks of four experts from a model file in the test, with a
window of 8 and a vocabulary of 32, and fit one batch of four rows for 200 updates with AdamW at
3e-3. Trained with the `next_token` objective alone, the report holds 200 steps and the last loss
is below half the first. The last metrics hold no `expert_usage_min`.

### TheBalanceTermKeepsEveryExpertInUse

Trained with a `sum` objective that adds the balance term, the report again holds 200 steps and a
last loss below half the first. `expert_usage_min` is above 0.10, at most 0.25 plus 256 times the
float unit roundoff, and `expert_usage_max` is at least 0.25 minus the same bound.

## Small_DecoderFromFiles and Medium_DecoderFromFiles

The cases build the decoder that `decoder_model.hpp` states as a model file: a vocabulary of 32, a
window of 8, a width of 16 and two blocks, with a head tied to the token table. The graphs are
traced for a batch of 2.

### TheTiedHeadSharesTheTokenTable

After a probe trace the model has the parameters `0.weight`, `1.weight` and `4.norm.weight` once
each and no `4.linear.weight`. The parameter element count equals the count worked out by hand:
`V * D + T * D + blocks * (12 * D^2 + 13 * D) + 2 * D`. The case is `Small_`.

### TheStateOfATiedHeadHoldsNoTensorOfItsOwn

The state created on the engine refuses `4.linear.weight` as `InvalidDescription`. The case is
`Medium_` because the state allocates tensors.

### TheTrainingGraphBindsEveryParameterOnce

The training graph of the next-token objective lists as many parameter bindings as the model has
parameters, and exactly one of them is named `0.weight`. `bind_parameters` pairs one slot with each
binding, so the count of the slots is the count of this list. The case is `Small_`.
