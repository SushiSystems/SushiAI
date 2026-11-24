# tests/layers/io/test_torch_writer.cpp

Suites: `Small_TorchWriter`.

Tests `IO::write_torch_state_dict`, in `modules/layers/io/include/SushiAI/io/torch_writer.hpp`, against `IO::open_torch_source`.

## TheSourceReadsBackADictionaryAtTheRoot

Three float32 tensors are written with no state key. The source lists them in order, with the name and shape given, and reads each value back.

## TheSourceReadsBackADictionaryUnderAStateKey

The same three tensors are written under a state key and read back the same way.

## WhatItCannotWriteIsRefused

An integer tensor and a folder name with a slash fail with `InvalidDescription`.
