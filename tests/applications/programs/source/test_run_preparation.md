# tests/applications/programs/source/test_run_preparation.cpp

Suite: `Medium_RunPreparation`, in `sushiai_programs_tests`.

Tests `prepare_run` of `run_preparation.hpp` over the `tiny_yolo` training file.

## AFileWithNoWeightsStartsAtItsSeedBeforeTheFirstBatch

A file that states neither `weights` nor `resume` builds a `PreparedRun` that starts at the seed of
the file, 7, at batch 0, with the optimizer at step 0.
