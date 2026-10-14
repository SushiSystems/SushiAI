# Regression tests

The functional regression tier holds the tests for guarantees that fail without a symptom: the run
still converges and still reports success. This document carries the reasoning the comments in
these files cite. Each section is named after one test file, and within it a heading names the
test or helper its paragraphs belong to.

## Shared conventions

### The SaTest alias

Inside a TEST_F body the fixture's base class injects the name `Test`, so `Test` alone resolves to
`::testing::Test`. Each file that uses `TEST_F` therefore declares `namespace SaTest = SushiAI::Test`, which keeps the
helpers in `SushiAI::Test` reachable from a test body.

