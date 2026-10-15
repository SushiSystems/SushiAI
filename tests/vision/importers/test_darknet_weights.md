# tests/vision/importers/test_darknet_weights.cpp


Suites: `Unit_DarknetWeights`.

Tests `read_darknet_header` over byte streams built in memory and `read_darknet_weights` over
files the cases write under the system's temporary directory. The sample file holds a 20-byte
header and index-valued floats for two convolutions, a plain one on layer 0 and a normalised
one on layer 2.

## ReadsAVersionZeroTwoHeaderWithA64BitSeen

A header of major 0, minor 2 and revision 5 reads an eight-byte seen count, and the stream
stands at byte 20.

## ReadsAVersionZeroOneHeaderWithA32BitSeen

A header of major 0 and minor 1 reads a four-byte seen count, and the stream stands at byte
16.

## ReadsAMajorOneHeaderWithA64BitSeen

A header of major 1 and minor 0 reads an eight-byte seen count, and the stream stands at byte
20.

## AcceptsAMajorAtTheLimitOfOneThousand

A header of major 1000 is read.

## RefusesAMajorAboveOneThousandAsUnsupported

A major of 1001 fails with `Unsupported` and a message holding "1001".

## RefusesAMinorAboveOneThousandAsUnsupported

A minor of 1001 fails with `Unsupported` and a message holding "1001".

## RefusesAHeaderThatEndsEarly

A stream that ends two bytes into the revision fails with `InvalidFile` and a message holding
"revision".

## ReadsEveryVectorInFileOrder

With floats 0 to 21 in the file, the plain convolution's biases start at 0 and its weights at
2. The normalised one's biases start at 4, its scales at 7, its rolling means at 10, its
rolling variances at 13 and its weights at 16. The header reads major 0 and minor 2.

## KeepsTheBatchNormVectorsOnlyForNormalisedLayers

The plain convolution has empty scales, means and variances and two weights. The normalised
one has three of each vector and six weights.

## DecodesEveryBitPatternAsItIs

Biases of negative zero and the smallest subnormal, and weights of a NaN with payload 1 and of
1.0, come back with the bits 0x80000000, 0x00000001, 0x7FC00001 and 0x3F800000.

## RefusesOneFloatShortWithTheLayerAndBothByteCounts

A file one float short fails with `InvalidFile`, "104 bytes" and
"convolution layer 2 needs bytes up to 108".

## RefusesOneFloatOverWithTheLayerAndBothByteCounts

A file one float over fails with `InvalidFile`, "112 bytes", "108 bytes" and
"4 bytes are left over after convolution layer 2".

## RefusesBytesAfterAHeaderWithNoConvolution

A file with one float after its header and an empty shape list fails with `InvalidFile`,
"no convolution is listed" and "4 bytes are left over".

## RefusesAShapeThatIsNotPositive

A shape with zero filters fails with `InvalidDescription`, "convolution layer 0" and
"filters 0".

## RefusesAShapeWhoseCountOverflows

A shape of 2^40 filters over 2^40 channels fails with `InvalidDescription`,
"convolution layer 0" and "too large to read".

## ReportsAMissingFileAsFileNotFound

A path that does not exist fails with `FileNotFound` and "could not open".
