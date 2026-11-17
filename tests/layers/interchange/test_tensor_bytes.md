# tests/layers/interchange/test_tensor_bytes.cpp

Suites: `Small_TensorBytes`.

Tests `TensorBytes` of `tensor_bytes.hpp`, the bytes of one foreign tensor as a view into the file or
as an owned copy. The file needs no device, no engine and no file.

## ADefaultTensorHoldsNoByte

A default tensor has size 0 and an empty byte range.

## ViewingKeepsTheCallersBytesWithoutACopy

A view of three bytes at offset 1 of a five-byte file reports the address of the file's second byte,
size 3 and offset 1, and says it is a view.

## OwningHoldsTheGivenBytesAndHasNoOffset

An owned tensor is not a view, holds the five bytes it was given in order and reports offset 0.

## OwningTakesTheBytesOutOfTheCallersVector

After the caller's vector is cleared, the owned tensor still holds its five bytes.
