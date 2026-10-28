# tests/core/eager/test_scope_trace.cpp


Suites: `Small_ScopeTrace`.

Tests how ScopeTrace cuts a trace into segments and which calls it rejects.

This file needs no device. A `ScopeTrace` reads the node count of a graph, so every case
appends nodes to a host-side graph, opens and closes scopes around them and inspects what
`seal()` returns. `modules/core/eager/README.md`, under `segment.hpp` and `scope_trace.hpp`,
defines a segment and the names a scope may have.

The cases fall into two groups, in file order: the segments a trace yields, and the calls
`open()`, `close()` and `seal()` reject.

## trace_nodes

`trace_nodes` appends a given number of K_ADD nodes to a graph, each adding one input value to
itself. The input is added as a value with no K_LEAF node, so the node count rises by exactly the
number asked for and a case can state every range as a literal.

## segment_is

`segment_is` compares one segment with an expected name, occurrence, begin and end. On a
mismatch it prints all four fields of both, so a failure shows which field differs.

## NodesOutsideAnyScopeFormOneUnnamedSegment

Three nodes traced with no scope open give one segment: empty name, occurrence 0, nodes 0 to 3.

## AScopeSplitsTheTraceIntoThreeSegments

One node, then two in a scope "body", then three give three segments. The unnamed run after the
scope has occurrence 1, because the unnamed run before it was occurrence 0.

## ANestedScopeIsNamedByItsPath

Inside "outer" and "inner" the depth is 2 and the segment is named "outer/inner". The nodes of
"outer" before and after it form two segments named "outer", with occurrences 0 and 1.

## AScopeOpenedTwiceGetsRisingOccurrences

A scope "cell" opened three times in a loop, with two nodes each time, gives three segments
named "cell" with occurrences 0, 1 and 2 and consecutive ranges.

## AnEmptyScopeLeavesNoSegment

A scope "cell" that closes with no node traced leaves no segment. The next "cell", which holds
one node, has occurrence 0, so the dropped scope did not use up an occurrence.

## SealWithAScopeStillOpenIsFatal

`seal()` stops the process while "outer" and "inner" are open, and the report contains "seal()",
"still open" and the path "outer/inner".

## CloseWithNoScopeOpenIsFatal

`close()` stops the process on a new trace, and again after one matched `open()` and `close()`.
Both reports contain "close()" and "no scope open".

## SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

## OpenWithAnEmptyNameIsFatal

`open("")` stops the process with a report containing "open()" and "empty scope name".

## OpenWithANameWithASlashIsFatal

`open("outer/inner")` stops the process. The report contains the name and "may not contain".

## OpenWithANameWithAHashIsFatal

`open("cell#0")` stops the process. The report contains the name and "may not contain".

## IsLegalScopeNameAgreesWithWhatOpenAccepts

`ScopeTrace::is_legal_scope_name` is true for "encoder" and "block 3", and false for the empty
name, "outer/inner" and "cell#0". A caller whose name comes from data tests it before `open()`.

## ASecondSealIsFatal

A second `seal()` stops the process with a report containing "seal()" and "sealed trace".

## OpenAndCloseAfterSealAreFatal

After `seal()`, `open("late")` and `close()` both stop the process with "sealed trace" in the
report.
