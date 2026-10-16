# tests/applications/programs/source/test_registries.cpp


Suites: `Small_Registries`, in `sushiai_programs_tests`.

Tests `compose_registries`, which fills the `Registries` of an app with the rows of the library
and then of each linked domain. No case needs a device.

## row_count

`row_count` adds the rows of every table a `Registries` holds: the nodes, layers and linked
nodes, the datasets, the objectives, the runs, the reports and the importers.

## TheLibraryRowsAreComposed

The composed registries find the layer "linear", the node "batchnorm2d", the dataset "rings"
and the objective "cross_entropy", and the run registry has a fallback.

## EachCompositionStartsFromEmptyRegistries

Two calls both succeed, the first holds rows, and the second holds as many as the first. A
composition that shared a table with an earlier one would fail on a duplicate row or hold more.
