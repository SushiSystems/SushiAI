# tests/language/umbrella/test_language.cpp


Suites: `Small_LanguageUmbrella`.

Tests that `modules/language/umbrella/include/SushiAI/language.hpp`, included alone with the
status assertions, reaches the core library and the language domain. The header only includes
others, so the cases use one symbol of each side. No case needs a device.

## TheUmbrellaRegistersTheLanguageRows

`NN::register_layers` and `Language::register_nodes` both succeed on one registry, which then
holds the "transformer_block" node row. `K_END_OF_TEXT` equals `K_BYTE_TOKEN_COUNT`.

## TheUmbrellaBuildsADomainCollator

`NextTokenCollator::make(4, 16)` succeeds. Through a `const Collator&` its task is
`K_TASK_NEXT_TOKEN` and its window length is 4.
