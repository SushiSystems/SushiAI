# tests/language/training/test_expert_balance_objective.cpp


Suites: `Small_LanguageTrainingExpertBalanceMake`, `Small_LanguageTrainingExpertBalanceLoss`,
`Small_LanguageTrainingExpertBalanceFactory`, `Small_LanguageTrainingExpertBalanceObjective`,
`Medium_LanguageTrainingExpertBalanceMetrics`, `Medium_LanguageTrainingExpertBalanceUsage`.

Tests `Language::Training::ExpertBalanceObjective` in
`modules/language/training/include/SushiAI/language/training/expert_balance_objective.hpp` and its factory
`make_expert_balance_objective`. The blocks the cases describe have four experts. The small
loss cases run the graph on the host evaluator (`SushiAI/graph/host_evaluator.hpp`); the medium
suites allocate `Float32` usage tensors of the shape [1, 4] on the engine.

## MakeRefusesWhatCannotBeBalanced

`make` fails with `InvalidDescription` for no balance ("at least one balance output", "0 and
1"), no usage ("1 and 0"), an empty balance name ("no name") and a usage of 0 experts ("'usage'
of 0"). The four inputs are rows. The getters of a made objective are not asserted; the factory
case below reads them through the factory.

## ItDeclaresNoLeafAndReadsNoRole

After `declare`, the leaves list no leaf and no required role, and the graph has no node.

## TheLossIsTheMeanOfTheBalanceOutputs

The outputs "balance2" and "balance3" are [1, 1] inputs that hold 2 and 4. The loss has the shape
[1]. Over "balance3" alone it evaluates to 4, and over both to 3. The graph is run, so the case
does not name the operations that build the loss. The two objectives are rows.

## TheFactoryReadsTheOutputsByTheirNames

Over a context that publishes "output", "balance2", "usage2", "balance3" and "usage3", the
factory builds an `ExpertBalanceObjective` whose balances are "balance2" and "balance3" and
whose usages are "usage2" and "usage3", the second of four experts.

## TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `weight` key ("objective states the key
'weight'"), for a context that publishes only "output" ("begin with 'balance'", "'usage'", "0
and 0", "among its 1") and for a "balance2" output of the shape [1, 2] ("'balance2'", "one
element", "[1, 2]"). The three descriptions are rows.

## MeasureReportsTheSmallestAndLargestShare

The step publishes "usage2" with four shares of 0.25 and "usage3" with the shares 0.125, 0.5,
0.0625 and 0.3125. `measure` writes two metrics, "expert_usage_min" of exactly 0.0625 and
"expert_usage_max" of exactly 0.5, each marked as a fraction. The shares are sums of powers of
two, so float32 holds them and the comparison needs no tolerance. The case stays medium for the
tensor allocation alone; the measure is host arithmetic.

## MeasureRefusesAUsageItCannotRead

Over a step that publishes only "usage2" with four shares, `measure` fails with
`InvalidDescription` for an objective that reads "usage9" ("'usage9'", "did not publish") and
with `Incompatible` for one that expects eight experts in "usage2" ("the usage of the
experts", "8 values", "holds 4"). Both are rows, and no metric is written in either.
