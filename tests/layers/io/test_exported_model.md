# tests/layers/io/test_exported_model.cpp

Suites: `Small_ExportedModel`.

Tests `IO::ExportedModel`, in `modules/layers/io/include/SushiAI/io/exported_model.hpp`, the files and notes an exporter returns. It holds no behaviour beyond its members.

## FilesKeepTheOrderTheyAreWrittenIn

Two files and one note are pushed. They read back in the order pushed, each with its name and byte count.
