# tests/language/blocks/test_decision_heads.cpp

Suites: `Small_BlockDecisionHeads`, `Small_BlockDecisionHeadsFactory`, `Medium_BlockDecisionHeads`.

Tests `Language::Blocks::DecisionHeads` and `make_decision_node` in
`modules/language/blocks/include/SushiAI/language/blocks/decision_heads.hpp`. Every row has 11 tokens,
a width of 8, two heads, one block, 3 options of 4 tokens and questions of 5 tokens. The numeric cases
run on an engine, in `Medium_`, for the reason `test_option_head.md` gives. The two probes work in
float64 over the states an encoder would produce, through `trace_heads`; the model run works in
float32 through `trace_decision`.

## TheFactoryBuildsANodeOfFiveInputsAndThreePorts

The node reads five values and has the ports `choice` of shape [3], `yes_no` of shape [1] and
`number` of shape [1], in that order.

## ATraceOfTheNodeAppendsThreeValuesOfTheStatedShapes

Tracing the node over 2 questions appends three values of shapes [2, 3], [2, 1] and [2, 1].

## TheNodeRefusesFourValues

Four values fail with `InvalidDescription` and leave the outputs empty.

## FlagsOfTheWrongShapeFailTheTrace

`trace_decision` over [2, 2] flags for 3 options returns the failure of the score.

## TheWalkNamesTheEncoderAndTheTwoHeads

The walk yields 18 entries: the 14 of the encoder under `dec.encoder`, then `dec.yes_no.weight`,
`dec.yes_no.bias`, `dec.number.weight` and `dec.number.bias`.

## TheFactoryRefusesARowThatDoesNotFit

The factory refuses a stray key, no `options`, an `options` that differs from the option tokens, no
`vocabulary`, a `length` below the longer text, four inputs, rank-2 question tokens, two question
lengths, rank-1 option tokens, two option lengths for 3 options and two option flags for 3 options.

Rows: StrayKey, NoOptions, OptionsAboveTheTokens, NoVocabulary, LengthBelowTheLongerText,
FourInputs, RankTwoQuestion, TwoQuestionLengths, RankOneOptions, TwoOptionLengths, TwoOptionFlags.

## TheScalarHeadsAreLinearInTheQuestionState

The yes-or-no logit and the number are w . q + b of the question state. Oracle: the sum worked on the
host in double precision from the eighths each head holds, the weight and bias of `yes_no` from offsets
0 and 1 and those of `number` from 2 and 3, compared through the tolerance helper with a bound of
eight terms and the bias.

## TheChoiceDifferentiatesAtTheQuestionAndTheOptions

The full-Jacobian gradient check of a cross-entropy over the choice logits at the question state and
the option states, with one padded option in the second question.

## TheYesNoLogitDifferentiatesAtTheQuestion

The full-Jacobian gradient check of a cross-entropy over the yes-or-no logits of the two questions,
taken as one row of two classes, at the question state.

## ThePaddedOptionsOfAModelHoldTheFill

The whole model holds `K_MASKED_ATTENTION_SCORE` at each padded option, a finite score at each held
one, and the argmax of each question is a held option.

## TheScalarHeadsReadNoOption

Two runs that differ in the option texts alone give the same yes-or-no logits and numbers bit for bit,
and a different choice.
