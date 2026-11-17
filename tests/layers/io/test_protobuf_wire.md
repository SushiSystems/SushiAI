# tests/layers/io/test_protobuf_wire.cpp

Suites: `Small_ProtobufWire`, `Small_ProtobufWireVarint`, `Small_ProtobufWireZigzag`,
`Small_ProtobufWireRefusal`.

Tests `IO::ProtobufWire`, in `modules/layers/io/include/SushiAI/io/protobuf_wire.hpp`, on bytes
written by hand in the test. The expected numbers come from the wire-format rules: a varint is
base-128 groups of seven bits, low group first; a tag is `(number << 3) | wire type`; a zigzag
value `n` decodes to `n / 2` for even `n` and `-(n + 1) / 2` for odd `n`. Message `K_GUIDE_MESSAGE`
is the example of the protobuf encoding guide: field 1 = 150, field 2 = "testing", field 3 holding
field 1 = 150, and field 4 packed as 3, 270, 86942.

## ReadsTheNumberItEncodes

Rows over the varint edges: 0, 1, 127, the two-byte 128, 300, 150, the largest 64-bit value in ten
bytes, and a redundant encoding of 1 in three bytes.

## ReadsTheSignedNumberItEncodes

Rows over zigzag: 0, -1, 1, -2, the largest and smallest 32-bit values, and the smallest 64-bit
value.

## ReadsAFixed32LowByteFirst, ReadsAFixed64LowByteFirst

`78 56 34 12` reads as 0x12345678 and the reader moves four bytes on; the eight bytes
`ef cd ab 89 67 45 23 01` read as 0x0123456789ABCDEF and the reader moves eight bytes on.

## TurnsFixedBitsIntoTheFloatsTheyEncode

`0x3F800000` is 1.0f, `0x3FF0000000000000` is 1.0 and `0xC0000000` is -2.0f. Each is exact, so the
comparison is equality.

## HandsOutLengthDelimitedBytesWithoutCopyingThem

The view of `03 61 62 63 64` holds 3 bytes and its pointer is the address of the second byte of the
caller's vector.

## ReadsAnEmptyLengthDelimitedValue

A length of 0 gives an empty view and the reader is at its end.

## SplitsATagIntoItsFieldNumberAndWireType

`12` is field 2, length-delimited; `95 01` is 149, field 18, fixed32.

## ReadsEveryKnownFieldOfTheGuideMessage

The guide message reads field by field, the packed field through `message()` and `read_varint()`,
and no field is skipped.

## SkipsUnknownFieldsOfEveryWireType

Fields 15 (varint), 16 (fixed64), 17 (length-delimited), 18 (fixed32) and 19 (a group holding a
varint field and a nested group) surround the guide message. The walk skips five fields, still reads
the known ones, and ends at the end of the bytes.

## ReadsAPackedFixed32Field

A packed field of 8 bytes reads as two fixed32 values, 1 and 0x102.

## ASubMessageContinuesTheOffsetsOfItsParent

A sub-message that starts at byte 4 of the file reports offset 4, and its parent reports offset 6.

## AFailureInASubMessageNamesTheByteInTheFile

A varint cut at the sixth byte of the file, inside a sub-message that starts at byte 4, fails with
`InvalidFile` naming byte 5 and the origin.

## NamesTheByteItFailsAt

Rows over a truncated varint at the start and after two varints, a varint of eleven bytes, a tenth
byte of 2, a cut zigzag, a fixed32 and a fixed64 cut short, a length past the end (at the start and
after two varints), a length of 2^64 - 1, a cut length, field number 0, a field number of 2^29, and
wire types 6 and 7. Each fails with `InvalidFile` naming the origin, the byte and the cause.

## RefusesToSkipAValueThatIsCutShort

Skipping a fixed64 over three bytes fails with `InvalidFile` at byte 0.

## RefusesAGroupThatNeverEnds

A group whose bytes stop at byte 2 with no end tag fails with `InvalidFile`.

## RefusesAnEndGroupThatClosesNothing

An end-group tag with no open group fails with `InvalidFile` at byte 1.

## AnEmptyStreamIsAtItsEnd

An empty stream is at its end at offset 0.
