# tests/layers/io/test_protobuf_wire_writer.cpp

Suites: `Small_ProtobufWireWriter`.

Tests `IO::ProtobufWireWriter`, in `modules/layers/io/include/SushiAI/io/protobuf_wire_writer.hpp`, against `IO::ProtobufWire`, the reader.

## EveryScalarKindReadsBack

A varint (300 and the largest 64-bit value), a zigzag value, two fixed words, a float, a double and a string are written. Each field reads back with the number, wire type and value given.

## ANestedMessageAndAPackedFieldReadBack

A message inside a message and a packed float field read back, each with its length and its values.
