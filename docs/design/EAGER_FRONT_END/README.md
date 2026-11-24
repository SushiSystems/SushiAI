# Eager front end

**Status:** Open — no phase planned yet
**About:** How a SushiAI program traces its model again on every step and compiles and records again only the regions whose structure changed.
**Modules:** modules/core/eager

## Problem

A model whose structure can change from one step to the next cannot be compiled once and
replayed. `modules/core/eager` traces the model on every step, signs each region of
the trace, and compiles and records again only the regions whose signature changed.

## Decision

None ruled in this topic yet. The specification the module was built from is
`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`.

## Design

The module's units, its two lanes and its refusals are in `modules/core/eager/README.md`.

The phases of an eager session, one dynamic graph per stretch between two reads, are not built.
Their plan is `docs/agent/2026_11_24_EAGER_PHASES/PLAN.md`; the open work on them is
`sa backlog list --area EAGER_FRONT_END`.

## Phases

None.

## Open questions

- **The owner of `modules/core/eager`.** `sa design owner modules/core/eager` returns PROFILING,
  whose text covers the eager session's stage times and their zones (phase P2). This topic lists
  no module until the owner rules which of the two owns it.
