# tests/language/decision_models/test_decision_set.cpp

Suites: `Small_DecisionSet` and `Small_DecisionSetRefusals`.

Tests the synthetic set of `modules/language/decision_models/include/SushiAI/language/decision_models/decision_set.hpp`: the records a seed draws, the mix of types and the
rule each type follows. The cases use values only and need no device. The seeds are 11 and 13, and
`K_TEST_EXTENTS.options` is the most options a choice may hold.

## TheSameSeedDrawsTheSameRecords

`generate_decision_set` with one seed and one count returns equal records twice, and unequal records
for another seed.

## ASetOfThirtyHoldsTenOfEachType

The types take turns along the set before the shuffle, so a set of thirty holds ten choices, ten
yes-or-no questions and ten number questions.

## EveryRecordFollowsTheRuleOfItsType

Over 300 records: a choice has a key of the first 8 ids and a text of two ids, between 2 and 4
distinct options of one id each, and the option at its label is the one `right_option_of` names for
its key; a yes-or-no or a number question has three ids and the answer its rule gives. The rules of
the two scalar types are the functions the generator calls, so the next two cases fix them by hand.

## TwoPositiveTokensMakeAYes

`yes_no_answer_of` of three ids with two in the first half of the range is true, and with one it is
false. The oracle is the count of the ids by hand.

## TheNumberIsTheMeanOfTheTokenWeights

`number_answer_of` of three ids of 31, of three ids of 28 and of three ids of 25 is 1, 0 and -1: the
weights are `(id - 28) / 3`, which are exact in float32 at those three points.

## TheGeneratorRefusesTheSet

Rows: NegativeRecords, OneOption, NineOptions. Each returns `InvalidDescription`.
