# tests/core/eager/test_scope.cpp


Suites: `Small_Scope`.

Tests that `Scope` opens its name on a `ScopeTrace` and closes it when it ends.

This file needs no device. A guard acts on a trace over a host-side graph, and `depth()` shows
whether a scope is open. `test_scope_trace.md` covers how a trace names its segments and which
names it refuses.

## trace_node

`trace_node` appends one K_ADD node to a graph, adding an input value to itself, so the scope
open at that moment has a node and leaves a segment.

## TheScopeIsOpenWhileTheGuardLives

The trace's depth is 1 inside the guard's block and 0 after it.

## NestedGuardsCloseInnermostFirst

An inner guard inside an outer one gives depth 2, then 1 once the inner block ends. `seal()`
succeeds, which it does only when every scope is closed, and the two segments are named
"outer/inner" and "outer" in node order.

## AGuardCannotBeCopiedOrMoved

`Scope` is neither copy- nor move-constructible nor assignable, so no second object can close
the scope a guard opened.
