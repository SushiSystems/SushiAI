# tests/language/registration

The tests of the module `language/registration`, one file per header of `modules/language/registration/include/`. The suite prefix of a case gives its ctest label.

## test_registration.cpp

Suites: `Small_LanguageRegistration`, `Small_DecoderFromFiles`, `Medium_DecoderFromFiles`,
`Medium_Moe`, `Medium_MoeMixture`, `Medium_MoeRouterGradient`.

Tests the three registration entry points of the language module, `Language::register_nodes`,
`Language::register_datasets` and `Language::register_objectives`, that model files using
their rows build, and the mixture-of-experts block they build against a host computation. The
registry, model and training-graph cases are `Small_` and need no engine; the mixture, the
gradient, the fit and the state cases are `Medium_`. The cases are described in
`test_registration.md`.
