# Failure contract

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-11 | P2 | open | Confirm the error contract in a build | one `sa build` and the test run compile and pass the tree after the removal of `SushiAI::Error` and the `noexcept` pass, and `test_noexcept_surface.cpp` has a… |  | `docs/design/PROGRAMME/README.md` | 2026-10-28 |
| RW-12 | P2 | open | Observe that a dropped recording releases storage and scratch | a test counts the engine's storage and scratch before and after a failed `record_and_run()` and sees both back where they started |  |  | 2026-10-24 |
| RW-13 | P2 | unmeasured | Show that the fused and plain GEMM epilogues agree under `ForceReference` | a test runs a fused and a plain GEMM epilogue under `ForceReference`, and the comment of `DeviceFusionSelector` cites its result |  |  | 2026-10-25 |
| RW-14 | P2 | open | State what the gradient check's reference mode proves | the gradient check's documentation says the reference path shares the element functor with the SYCL path, so it separates launch and index errors, not formula… |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
