# tests/applications/programs/source/test_domains.cpp


Suites: `Small_Domains`, in `sushiai_programs_tests`.

Tests `register_vision_domain` and `register_language_domain`. Each adds the rows of its domain,
or nothing in a build that does not link the domain, so every case holds in both builds. No case
needs a device.

## row_count

`row_count` adds the rows of every table a `Registries` holds.

## EachDomainRegistersIntoEmptyRegistries

Each domain registers into a fresh `Registries` without a failure.

## TheDomainsOnlyAddBesideTheLibraryRows

A `Registries` is filled with the library's rows through `NN::register_layers`,
`Data::register_datasets`, `Train::register_objectives` and `register_runs`. Both domains then
register without a failure, so no domain row takes a name the library holds. The row count does
not fall, and the layer "linear", the dataset "rings", the objective "cross_entropy" and the
fallback run are still found.
