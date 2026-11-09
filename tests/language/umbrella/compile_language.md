# tests/language/umbrella/test_language/compile_language.cpp


A compile-only unit: it holds no test and the runner never lists it.

Tests that `modules/language/umbrella/include/SushiAI/language.hpp`, included alone, names the
core node registry and the collator of the domain. The file includes no other SushiAI header.
The header only includes others, so the names are what it pins. The rows that registered the
language nodes and built a collator moved out: `tests/language/registration/test_registration.cpp`
holds the registration and `tests/language/datasets/test_next_token_collator.cpp` the collator.

The file asserts at namespace scope that `NN::NodeRegistry` is default constructible, that
`Language::Datasets::Collator` is abstract and that `NextTokenCollator` derives from it. It needs
no device and runs nothing; the program compiles it.
