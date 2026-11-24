# tests/layers/io/test_protobuf_text_writer.cpp

Suites: `Small_ProtobufTextWriter`.

Tests `IO::write_protobuf_text`, in `modules/layers/io/include/SushiAI/io/protobuf_text_writer.hpp`, against the protobuf text parser.

## TheParserReadsBackTheTree

A tree of every field kind, an escaped string and a nested message is written and parsed. Each field keeps its name, kind, text and sub-fields.

## AFieldItCannotWriteIsRefused

A field with no name and a bare token that holds a space fail with `InvalidDescription`.
