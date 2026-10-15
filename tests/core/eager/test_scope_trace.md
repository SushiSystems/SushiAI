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

## SealWithAScopeStillOpenThrows

`seal()` throws while "outer" and "inner" are open, and the message contains "seal()", "still
open" and the path "outer/inner". The depth is still 2 afterwards.

The case then traces one more node, closes both scopes and seals. The result is one segment
"outer/inner" over nodes 0 to 2, which shows the failed `seal()` did not cut the running
segment.

## CloseWithNoScopeOpenThrows

`close()` throws on a new trace, and again after one matched `open()` and `close()`. Both
messages contain "close()" and "no scope open".

## SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

## OpenRejectsAnEmptyName

`open("")` throws with a message containing "open()" and "empty scope name", and the depth
stays 0.

## OpenRejectsANameWithASlash

`open("outer/inner")` throws. The message contains the name and "may not contain", and the
depth stays 0.

## OpenRejectsANameWithAHash

`open("cell#0")` throws. The message contains the name and "may not contain", and the depth
stays 0.

## ARejectedNameDoesNotSplitTheRunningSegment

One node is traced, `open("a/b")` throws, and one more node is traced. `seal()` returns one
unnamed segment over nodes 0 to 2. A trace that ended the running segment before it validated
the name would return two.

## ARejectedCloseDoesNotSplitTheRunningSegment

The same arrangement with a `close()` that throws for having no scope open also returns one
unnamed segment over nodes 0 to 2.

## ASecondSealThrows

A second `seal()` throws with a message containing "seal()" and "sealed trace".

## OpenAndCloseAfterSealThrow

After `seal()`, `open("late")` and `close()` both throw with "sealed trace" in the message, and
the depth stays 0.
