# tests/layers/io/test_zip_writer.cpp

Suites: `Small_ZipWriter`.

Tests `IO::ZipWriter`, in `modules/layers/io/include/SushiAI/io/zip_writer.hpp`, against `IO::ZipArchive`.

## TheReaderReadsBackEveryEntry

Three entries of different names and sizes are written, aligned to 1, 64 and 64. The reader finds each with its bytes. Each entry's data starts on its alignment, and the CRC-32 in its local header equals a bit-by-bit reference.
