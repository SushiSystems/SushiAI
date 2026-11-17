# tests/vision/training/compile_free_loss_terms.cpp

A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`FreeLossTerms` in `modules/vision/training/include/SushiAI/vision/training/free_loss_terms.hpp` is
trivially copyable and default constructible, since the host oracle, the training loop and the
report pass the terms of a step by value.

`FreeLossTerms` is plain data with no behaviour of its own: five doubles and their default value.
The values the oracle puts in it are checked by `tests/support/test_free_detection_loss_host.cpp`,
and the values the graph puts in it by `test_free_detection_loss.cpp`.

The file needs no device and runs nothing.
