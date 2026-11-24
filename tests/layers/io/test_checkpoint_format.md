# tests/layers/io/test_checkpoint_format.cpp

The checkpoint format's round trip, a version 1 file, the run-state section, and its refusal of each malformed byte pattern.

Suites: `Small_CheckpointFormat`, `Small_CheckpointFormatVersion`, `Small_CheckpointFormatOwner`,
`Medium_CheckpointFormatDeathTest`.

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `modules/layers/io/include/SushiAI/io/checkpoint_format.hpp` out of
`modules/layers/io/include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

## Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

## RoundTripsNamesShapesAndPayloads

The sample stream opens as two records. The first is `0.weight`, `Float32`, [2, 3], 24
payload bytes, and its payload reads back as the six floats written. The second is `0.bias`
of [2] with the two floats written. `read_count` is then 2.

## RejectsAForeignMagic

With the magic bytes replaced by "JUNK" the read fails with `InvalidFile`, "SACP" and
"not a SushiAI checkpoint".

## ARejectedVersionNamesTheVersionsRead

Run once for each row of `Small_CheckpointFormatVersion`: version 3 and version 0. Each is refused
as `InvalidFile`, naming the version and "versions 1 to 2".

## RejectsACountOverTheBound

With the record count patched to 4097 the read fails with `InvalidFile`, "4097" and "4096".

## RejectsATruncatedHeader

The first six bytes of the stream fail with `InvalidFile`, "ended while reading" and
"version".

## RejectsATruncatedPayload

The stream without its last four bytes fails with `InvalidFile`, "ended while reading" and
"0.bias".

## RejectsANameLengthOverTheBound

With the first name length patched to 257 the read fails with `InvalidFile`, "257" and "256".

## RejectsARankOverTheBound

With the first rank patched to 7 the read fails with `InvalidFile`, "0.weight", "rank 7" and
"6".

## RejectsANonPositiveDim

With the first extent patched to 0 the read fails with `InvalidFile`, "0.weight" and
"must be positive".

## RejectsAnUnknownDType

With the first element type patched to 99 the read fails with `InvalidFile`, "0.weight", "99"
and "does not know".

## RejectsACountTypeTheFormatDoesNotCarry

With the first element type patched to the `UInt32` value the read fails with `InvalidFile`,
"0.weight", "uint32" and "does not carry".

## RefusesToWriteACountTypeTheFormatDoesNotCarry

Writing a `UInt32` record fails with `InvalidDescription`, "counts", "uint32" and "does not
carry".

## RejectsMoreRecordsThanDeclared

A writer created for 2 records writes two. A third `write` is fatal with "declared 2" and
"'c'".

## RejectsFewerRecordsThanDeclared

A writer created for 2 records writes one. `finish` is fatal with "declared 2" and
"1 were written".

## RejectsATensorCountOverTheBoundAtConstruction

`CheckpointWriter::create` for 4097 records fails with `InvalidDescription`, "4097" and
"4096".

## ReadingPastTheDeclaredCountIsAnError

After both records of the sample stream are read, a third `next` is fatal with "declared 2".

## WritesTheVersionTwoHeaderByteByByte

The sample stream opens with the sixteen bytes 'S', 'A', 'C', 'P', then 2, 2 and 0 as
little-endian 32-bit values. Its size is those sixteen bytes plus 60 bytes for the weight
record and 34 for the bias record.

## AVersionOneFileReadsAsModelStateAlone

A version 1 stream opens with `version` 1, two records and a run-state count of 0, and both
payloads read back as written.

## RoundTripsARunStateSection

In the run-state stream, byte 50 opens the section with the owner length 3 and "sgd", and the
run-state count field holds 2. The reader reports version 2, one model record and two
run-state records. After the model record, `begin_run_state` returns "sgd". The next records
are `step_count`, `Int64` of [1] holding seven, and `0.bias.momentum` holding the two floats
written. `read_count` is 1 and `run_state_read_count` is 2.

## AReaderOfModelStateLeavesTheRunStateUnread

The owner length of the run-state section is set to zero, which makes the section malformed. A
reader of the model records alone still reads them, and only the reader of the run state refuses
the file, with "names no owner".

## RejectsARunStateCountOverTheBound

With the run-state count patched to 16385 the read fails with `InvalidFile`, "16385",
"16384" and "run-state". `CheckpointWriter::create` with 16385 run-state records fails with
`InvalidDescription` and the same three texts.

## AMalformedRunStateOwnerIsRefused

Run once for each row of `Small_CheckpointFormatOwner`: an owner of length zero, one of 257 bytes,
one cut short, and a writer given an empty owner. Each row states its code and its message.

## RejectsATruncatedRunStatePayload

The run-state stream without its last three bytes fails with `InvalidFile`,
"ended while reading" and "0.bias.momentum".

## TheRunStateOpensAfterTheModelRecords

On a writer that declared one model record and wrote none, `begin_run_state` is fatal with
"declared 1 records" and "0 were written". On a reader that has read none it is fatal with
"declared 1 records" and "0 were read".

## ARunStateThatWasNotDeclaredCannotOpen

On a writer created without run-state records, `begin_run_state` is fatal with
"declared no run-state record". On a reader of a version 1 stream, after both records are
read, it is fatal with "declares no run-state record".

## TheRunStateHoldsExactlyTheRecordsDeclared

On a writer that declared one run-state record, `finish` right after `begin_run_state` is
fatal with "declared 1 run-state records" and "0 were written". After one record is written,
a second `write` is fatal with "declared 1 run-state records" and "'b'".
