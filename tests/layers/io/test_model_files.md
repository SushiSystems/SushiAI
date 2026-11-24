# tests/layers/io/test_model_files.cpp

Suites: `Small_ModelFiles`.

Tests `IO::read_model_files`, in `modules/layers/io/include/SushiAI/io/model_files.hpp`, which reads the files `write_imported_model` wrote.

## ReadsTheFilesTheImportWriterWrote

A model with two weights, a training file and a data block is written to a scratch folder and read back. The model text, each weight's name, shape and bytes, the training keys and the data key are the ones written.

## AModelFileAloneStatesNothingElse

Only the model file is given. The read model has no weights, and its training and data attributes are null.

## AMissingWeightFileIsRefused

A weight path that names no file is refused.
