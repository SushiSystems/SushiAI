# tests/language/generation

The tests of the module `language/generation`, one file per header of `modules/language/generation/include/`. The suite prefix of a case gives its ctest label.

Every test file has a sidecar `<name>.md` that lists its suites and describes each case. A header
that is a port (`fed_leaves.hpp`, `one_position_step.hpp`, `window_logits.hpp`) has a
compile-only unit `compile_<name>.cpp` and its in-memory adapter has a `test_in_memory_<name>.cpp`.

| File | Tests |
| --- | --- |
| `test_sampler.cpp` | `RandomGenerator` and the `Sampler` contract over the four samplers |
| `test_greedy_sampler.cpp`, `test_temperature_sampler.cpp`, `test_top_k_sampler.cpp`, `test_nucleus_sampler.cpp` | One sampler each |
| `test_sampler_section.cpp`, `test_generation_task.cpp` | The readers of the sampler section and of the task section |
| `test_generator.cpp` | The `Generator` interface and its contract over the three generators |
| `test_window_generator.cpp`, `test_cached_generator.cpp` | The two loops |
| `test_position_feed.cpp`, `test_ring_position_feed.cpp`, `test_table_position_feed.cpp` | The choice of a position feed and the two feeds |
| `test_in_memory_fed_leaves.cpp`, `test_in_memory_one_position_step.cpp`, `test_in_memory_window_logits.cpp`, `test_in_memory_generator.cpp` | The in-memory adapters |
| `compile_fed_leaves.cpp`, `compile_one_position_step.cpp`, `compile_window_logits.cpp` | Compile-only checks of the three ports |

Two support headers hold what several files share. `generation_test_support.hpp` holds the counted
draws and the compared shares of the samplers; the bound of a share is statistical and its
derivation is beside `K_SHARE_TOLERANCE`. `position_feed_test_support.hpp` holds the fed leaves,
the host engine and the in-memory table step of the feeds and of the cached generator.

A `Medium_` case builds a decoder from a model file and runs it on the device (the two loops
against each other), or starts a child process (a death test). Everything else runs `Small_`, on
an in-memory adapter or on a host engine.
