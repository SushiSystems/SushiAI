# Security Policy

We take the security of SushiAI seriously. This document explains how to
report a vulnerability and what to expect in return.

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security problem.** Public
disclosure before a fix is available puts every user at risk.

Instead, report it privately through one of:

- **Email (preferred):** **mustafagarip@sushisystems.io**
- **Discord:** reach the maintainer privately on the community server —
  invite: `https://discord.gg/y5639YuaPh`

If you prefer, you may also use GitHub's **private vulnerability reporting**
("Report a vulnerability" under the repository's Security tab).

### What to include

A good report lets us reproduce and assess quickly:

- A clear description of the issue and its impact.
- The affected component and the version / commit hash.
- Step-by-step reproduction, ideally with a minimal proof of concept.
- The platform, toolchain, and build configuration.
- Any suggested mitigation, if you have one.

## What to expect

- **Acknowledgement** within **72 hours**.
- An initial assessment and severity classification within **7 days**.
- Regular updates while we work on a fix, and credit in the release notes once a
  fix ships (unless you ask to remain anonymous).

Please give us a reasonable window to release a fix before any public
disclosure. We are happy to coordinate timing with you.

## Scope and threat model

SushiAI is at **AI-0** — a scope-reset skeleton with one compiled symbol
(`SushiAI::version_string()`) and no operations of its own yet. Nearly all
compute-relevant security surface today actually lives in
[SushiBLAS](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/SECURITY.md)
(the kernels) and
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime/blob/main/docs/SECURITY.md)
(the scheduler and memory pool) — please report a bug specific to either of
those directly to that repository. This document will grow real,
SushiAI-specific scope statements as AI-1 onward lands an autograd core,
graph API, and (eventually) model/checkpoint loading — the last of which,
like SushiBLAS's tensor file loaders, will be a genuine untrusted-input
surface worth its own explicit scope.

In scope today:

- Memory-safety bugs in SushiAI's own code (`src/`, `include/SushiAI/`) —
  currently a very small surface, but a real one.
- CMake/packaging defects that leak SushiAI's private build flags to a
  `find_package(SushiAI)` consumer (see `tests/package/`).

## Supported versions

SushiAI is pre-1.0 and under active development. Security fixes are applied
to the latest `main`; there is no long-term support branch yet.

## Contact

For anything that isn't sensitive, a GitHub issue or Discussion is fine. For
security matters, use the private channels above or email
**mustafagarip@sushisystems.io**.
