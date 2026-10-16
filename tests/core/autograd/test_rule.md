# tests/core/autograd/test_rule.cpp


Suites: `Small_AutogradRule`.

Tests the rule types of `modules/core/autograd/include/SushiAI/autograd/rule.hpp` and
`concat_rules`, which joins two rule tables at compile time. The rules in the tables are the
file-local `first_rule` and `second_rule`, which do nothing; only their addresses are compared.

## K_JOINED

`K_JOINED` is `concat_rules` of `K_LEFT`, the rows K_ADD and K_SUB with `first_rule`, and
`K_RIGHT`, the row K_MUL with `second_rule`. It is a `constexpr` table, so the join runs during
compilation.

## ARuleIsAFunctionOfAReadOnlyTape

`BackwardRule` is a pointer to a function of a `const GradientTape&`, and a `RuleEntry` keeps
the operation and the rule it was given.

## TheJoinedTableHoldsTheLeftRowsThenTheRight

`K_JOINED` has three rows: K_ADD and K_SUB with `first_rule`, then K_MUL with `second_rule`. Two
`static_assert` lines check the size and the last row during compilation.

## JoiningAnEmptyTableKeepsTheOtherOne

Joining an empty table on the left gives the one row of `K_RIGHT`, and joining one on the right
gives the two rows of `K_LEFT` in order.
