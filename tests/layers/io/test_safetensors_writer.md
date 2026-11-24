# tests/layers/io/test_safetensors_writer.cpp

Suites: `Small_SafetensorsWriter`.

Tests `IO::write_safetensors`, in `modules/layers/io/include/SushiAI/io/safetensors_writer.hpp`, against `IO::SafetensorsReader`.

## TheReaderReadsBackEveryTensor

A float32, a float64 and a half tensor are written with seeded bytes. The data starts on an 8-byte boundary, and the reader gives back each tensor's name, type, shape and bytes.

## AnIntegerTensorIsRefused

An integer tensor fails and the message names it.
