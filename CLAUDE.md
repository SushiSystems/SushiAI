# CLAUDE.md

## Getting oriented

When starting work on this project:

1. Follow the Sushi Systems doctrine (`sushiskills`) and repository conventions.
2. Read `docs/CONTRIBUTING.md` first and follow its rules.
3. Read `docs/README.md` (the manual index) and `docs/architecture/ARCHITECTURE.md` before modifying subsystems.
4. `docs/design/` holds design specifications and roadmaps. Read the relevant design document before changing a subsystem. Never edit generated Doxygen output under `docs/api-site/`.

## Active skills

The repository enforces these skills:

- `sushiskills`: Stack-wide doctrine, SOLID principles, maintainability priority, and CLI-only entry point.
- `repository-layout`: Standard directory layout, module isolation, and root hygiene.
- `cpp-code-style`: C++17 standard, LLVM base, 4-space indentation, 100-column limit, and Allman braces.
- `documentation`: Single-home documentation structure, agent work lifecycle, and changelog discipline.
- `project-tools`: Verification scripts and checkers under `tools/`.
- `source-comments`: Doxygen formatting, license blocks, and file headers.
- `logging`: Log categories, levels, and structured emission.
- `continuous-integration`: CI workflows, build verification, and release gates.
- `humanizer`: Plain, factual prose in documentation, comments, and reports without generated filler.

## How to work

Act as an expert in whatever role the task requires, and work with real engineering discipline: no quick hacks that happen to pass.

### Priority 1: Maintainability

Everything must be SOLID and encapsulated within its own boundaries. Every module must be isolated so that:

- Changes stay local instead of rippling across the codebase.
- A reader finds any symbol quickly.

Maintainability determines the long-term fate of the project; it precedes all other priorities, including performance.

### Priority 2: Performance

Treat wasted cycles as a concrete cost in money and energy: "engineer it once carefully, run it a thousand times cheaply." Profile and optimize algorithms up front so runtime execution remains efficient.

### Quality bar

Hold an obsessive standard of quality and correctness. "Good enough for now" work is unacceptable.

### Honesty

Be direct and factual. Mutual honesty is required for progress.

If a task exceeds available scope, never cut corners and claim completion. State plainly what is complete and what remains open.

If a task requires an architectural or policy decision, ask the user instead of guessing or picking arbitrarily.

### Building and toolchain

Never invoke CMake, Ninja, or CTest directly. All builds, tests, and tool invocations must run through the `sa` CLI (`sa build`, `sa test`, `sa run`, `sa doxygen`). Only the CLI configures the necessary environment and runtime library paths.

### Task tracking

Create actionable tasks up front and update them as work moves forward. Track progress through that plan rather than memory.

---

love you boss, kolay gelsin.
