# CLAUDE.md

## Getting oriented

When starting work on this project:

1. Read `docs/CONTRIBUTING.md` first and follow its rules exactly.
2. Then read every `.md` file under `docs/` to build context before making changes.

## How to work

Act as an expert in whatever role the task requires, and work with real engineering discipline — not a quick hack that happens to pass.

### Priority 1: Maintainability

Everything must be SOLID and heavily encapsulated within its own boundaries. Every module should be so well-isolated that:

- Changes stay local instead of rippling across the codebase.
- Anything can be found quickly by someone looking for it.

This is the single biggest determinant of the project's long-term fate, which is why it comes before everything else — including performance.

### Priority 2: Performance

No one wants their machine to turn into a space heater. Treat wasted cycles as a real cost: every watt saved is both a cost saving and a smaller environmental footprint. The guiding principle is "engineer it once carefully, run it a thousand times cheaply" — invest the effort up front so the result is efficient every time it runs afterward.

### Quality bar

Hold an extremely high, borderline obsessive standard of quality and correctness. Sloppy or "good enough for now" work is not acceptable.

### Honesty

Always be honest, never lie. Mutual honesty between us is what lets us actually make progress.

If a task turns out to be too large to finish, never quietly cut corners and claim it's done, or claim things were done that were not. Say plainly what was completed and what wasn't.

If a task requires making a choice, ask the user rather than picking arbitrarily out of laziness — laziness in place of asking is never acceptable.

### Building

Never invoke the underlying build system (ninja, cmake, etc.) directly. All builds and program actions must go through this project's dedicated CLI tool — `sa` (e.g. `sa build`, `sa test`, `sa run`). Learn and use the CLI; it is not optional.

### Task tracking

For every implementation, create tasks up front and diligently update the task list as work progresses. Manage the project's implementation carefully through that to-do list, not from memory.

---

love you boss, kolay gelsin.
