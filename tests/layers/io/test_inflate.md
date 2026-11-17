# tests/layers/io/test_inflate.cpp

Suites: `Small_InflateBlocks`, `Small_InflateOutputBounds`, `Small_InflateCorruption`,
`Small_Inflate`.

Tests `IO::inflate`, in `modules/layers/io/include/SushiAI/io/inflate.hpp`. The streams are byte
arrays that Python's `zlib` made, each beside the exact line that made it, and the texts they decode
to are known from that line: the oracle is the compressor's own input. Four streams are written by
hand from RFC 1951 for the refusals, each with the bits spelled in a comment. The bit offsets of the
refusals are counted from the format: three header bits, then the bits each field takes.

## InflateReproducesTheBytesTheCompressorWasGiven

Rows over a stored block, an empty stored block, a fixed-Huffman block, a dynamic-Huffman block of
the text `0,1,...,59,` and a stream of three blocks (a fixed block, the empty stored block that
`Z_FULL_FLUSH` writes, a last fixed block). Each decodes to exactly the text the compressor was given.

## InflateRefusesAnOutputTooSmallForTheStream

Rows over the stored, fixed, dynamic and three-block streams, each into an output one byte short of
its text. Each fails with `InvalidFile`, naming the stream and "writes past the end of the output".

## InflateRefusesACorruptStreamAtItsBitOffset

Rows over a block of the reserved type 3 (bit 3), a stored block whose length 5 and check 0 are no
complements (bit 40), a code-length code of four symbols of length 1 that has too many codes (bit
29), a code-length code of one symbol of length 1 that leaves half its patterns unused (bit 29), a
match at distance 1 before any byte was written (bit 15), a dynamic stream cut after ten bytes
(bit 80) and an empty stream (bit 0). Each fails with `InvalidFile`, naming the stream, the bit
offset and the reason.

## InflateReturnsTheBytesWrittenForALargerOutput

The fixed stream into an output of 100 bytes returns 23, the length of its text, and the text is at
the front of the output.

## InflateStopsAtTheLastBlock

Two bytes after the last block of the stored stream are left unread: the 18 bytes of its text are
returned.
