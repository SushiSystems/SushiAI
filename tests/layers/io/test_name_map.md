# tests/layers/io/test_name_map.cpp

Suites: `Small_NameMap`.

Tests `IO::NameMap`, in `modules/layers/io/include/SushiAI/io/name_map.hpp`, on JSON text held in
the test. Nothing reads a file from disk; `NameMap::read` adds only `parse_json_file` to `parse`.

## ReadsTheStateKeyTheRulesAndTheDrops

A map with `state_dict` "model", two rename rules (`backbone.` to `body.`, then the empty prefix to
`m.`) and one drop entry. The key reads back as `model`, the rules in file order with the views
`RenameRule` states. `renamed` over them sends `backbone.x` to `body.x`, because the first match
wins, and `head.x` to `m.head.x`, because the empty prefix matches every name.

## AnEmptyObjectChangesNothing

`{}` has no state key, no rules and drops nothing.

## ADropEntryMatchesTheNameOrADottedSuffix

With `num_batches_tracked` and `bn.total` on the list, the whole name and any name that ends in a dot
and the entry are dropped (`layer1.0.bn1.num_batches_tracked`, `layer1.bn.total`). A name whose last
component only ends in the entry (`x_num_batches_tracked`, `xbn.total`) and a name that continues
past the entry (`num_batches_tracked.weight`) are kept.

## RefusesARootThatIsNoObject

`[1]` fails with `InvalidDescription`, naming `map.json` and "not an object".

## RefusesAnUnknownKey

`renames`, a misspelling of `rename`, fails with `InvalidDescription` naming the key.

## RefusesAnEmptyStateKey

`"state_dict": ""` fails with `InvalidDescription`: the key is left out for a file that is the
dictionary.

## RefusesARenameRuleThatIsNotFromAndTo

A rule with no `to`, a rule with a third key `x` and a rule whose `to` is the number 3 each fail with
`InvalidDescription`, naming the rule by its index and the field at fault.

## RefusesAnEmptyDropEntry

The entry at index 1 of `["a", ""]` fails with `InvalidDescription`, naming the index and "empty".

## RefusesTextThatIsNoJson

`{"drop":` fails with `InvalidFile`, naming `map.json`.
