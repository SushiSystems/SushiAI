# tests/core/darknet_reference/test_darknet_reference.cpp


Suites: `Medium_DarknetReference`, `Medium_DarknetReferenceEnvironment` and `Small_DarknetTensor`.

Tests `find_darknet_file` and `find_darknet_reference` over a folder the cases write into, and
`decode_darknet_tensor` over text and bytes held in memory.

The `Medium_DarknetReference` fixture writes into a folder under the temporary directory and
passes it to the lookups, which makes those cases medium. `Medium_DarknetReferenceEnvironment`
removes `SA_REFERENCE_DATA_DIR` and restores it afterwards, for the one case that reads the
variable. Every reference lives in the run folder "tiny_run". The tensor cases build the
little-endian float32 bytes in the test and are small.

## AFileInTheDirectoryIsFound

"model.cfg", written directly in the folder, is found at its path there.

## AMissingFileNamesItsPath

"absent.cfg" is not found, and the reason names it.

## AnUnsetVariableIsTheReason

With the variable removed, "model.cfg" is not found and the reason names
`SA_REFERENCE_DATA_DIR`.

## AReferenceNeedsBothFiles

A reference with a `.json` and no `.bin` is not found, and its reason names "conv.bin". Loading
it fails with `FileNotFound` naming the same file.

## AFeatureMapIsMovedToChannelsLast

A CHW tensor of shape [2, 1, 3] holding 0, 1, 2 and 10, 11, 12 loads with the extents [1, 3, 2]
and the values 0, 10, 1, 11, 2, 12.

## ATensorWithoutAnOrderIsKeptAsStored

A [2, 2] tensor with no "order" keeps its extents and its values as written.

## AByteCountThatDiffersFromTheShapeIsRefused

A shape of [3] over a payload of two values fails with `InvalidFile`, naming the bytes.

## ADtypeOtherThanFloat32IsRefused

A "dtype" of "float16" fails with `InvalidFile`, naming the key and the value.
