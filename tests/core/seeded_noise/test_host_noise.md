# tests/core/seeded_noise/test_host_noise.cpp


Suites: `Small_HostNoise`.

Tests that `HostNoise` is a reproducible stream whose values stay within their scale.

This file needs no device. The file-local `draw` returns the first values of a stream, each
from `next`; the cases draw 1000 values.

## TheSeedAloneFixesTheStream

Two streams seeded 42 are equal, and a stream seeded 43 differs from them.

## ValuesStayWithinTheScale

With a scale of 0.25 every value of the stream seeded 7 lies in [-0.25, 0.25), and the values
hold both signs.

## FillWritesTheValuesNextReturns

`fill` of 16 values at a scale of 2 writes the same values, in the same order, as `next` on a
stream of the same seed.
