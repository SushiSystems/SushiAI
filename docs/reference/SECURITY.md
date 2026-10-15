# Security Policy

We take the security of SushiAI seriously. This document explains how to
report a vulnerability and what to expect in return.

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security problem.** Public
disclosure before a fix is available puts every user at risk.

Instead, report it privately through one of:

- **Email (preferred):** **mustafagarip@sushisystems.io**
- **Discord:** reach the maintainer privately on the community server:
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

SushiAI is at **AI-1**: a graph IR, a differentiation transform, layers,
optimizers, a data seam and a training loop, none of which contain a kernel:
SushiAI writes no SYCL, and every compute call goes through
`SushiBLAS::Engine` from one file (`src/graph/lowering.cpp`).

The one place SushiAI itself reads bytes it did not write is the MNIST IDX
loader (`src/data/dataset.cpp`). It checks both magic numbers, requires the
image and label counts to agree, and treats a short read as an error, but it
does **not** bound the declared `rows * cols` before sizing its host buffer,
so a malformed header can turn into an allocation failure rather than a clean
rejection. It is aimed at a corrupted download, not at a hostile file; do not
point it at input you do not control. Nearly all other compute-relevant
security surface still lives in
[SushiBLAS](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/SECURITY.md)
(the kernels) and
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime/blob/main/docs/SECURITY.md)
(the scheduler and memory pool): please report a bug specific to either of
those directly to that repository. This document will grow further scope
statements as model/checkpoint loading lands: that, like SushiBLAS's tensor
file loaders, will be a genuine untrusted-input surface worth its own
explicit section.

In scope today:

- Memory-safety bugs in SushiAI's own code (`src/`, `include/SushiAI/`).
- Malformed-input handling in the MNIST IDX loader (`src/data/dataset.cpp`),
  within the limits stated above.
- CMake/packaging defects that leak SushiAI's private build flags to a
  `find_package(SushiAI)` consumer (see `tests/package/`).

## Supported versions

SushiAI is pre-1.0 and under active development. Security fixes are applied
to the latest `main`; there is no long-term support branch yet.

## Contact

For anything that isn't sensitive, a GitHub issue or Discussion is fine. For
security matters, use the private channels above or email
**mustafagarip@sushisystems.io**.
