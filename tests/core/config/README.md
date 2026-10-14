# tests/core/config

The tests of the module `core/config`, one file per header of `modules/core/config/include/`. The suite prefix of a case gives its ctest label.

## test_json_events.cpp

Suites: `Unit_JsonEvents`.

Tests `Config::read_json_events`. Each case writes a file under the system's temporary
directory and removes it at its end. A local `EventLog` implements `Config::JSONEvents`, keeps
every event as one line of text and can return false at a chosen event.

### EventsArriveInTheOrderOfTheText

A document with an object, an array, two whole numbers, a real, a string, both booleans, a null
and a nested object gives 16 events in the order of the text. Keys arrive as `key` events
before their values.

### AWholeNumberAboveInt64IsAReal

In `[9223372036854775807, 18446744073709551615]` the first entry arrives as
"whole 9223372036854775807" and the second as a real. The read gives four events.

### ADocumentLongerThanTheBufferArrivesWhole

An array of 40000 whole numbers, whose text is longer than twice
`K_JSON_EVENT_BUFFER_BYTES`, gives 40002 events. The last number is "whole 39999" and the last
event is the end of the array.

### AHandlerThatStopsEndsTheReadWithSuccess

A log that returns false at its fourth event receives four events from `{"a": [1, 2, 3]}`, and
the read is ok.

### TextThatIsNoJSONIsRefusedWithItsOffset

Three files fail with `InvalidFile` and "not valid JSON": a document cut short, which also
names the file and "at byte 12", a document with trailing text ("at byte 5") and an empty file
("at byte").

### AnAbsentFileIsRefusedByItsPath

A path with no file fails with `FileNotFound` and a message containing the file name and "could
not open".

## test_json_writer.cpp

Suites: `Unit_JsonWriter`.

Tests `Config::write_json`. A depth of 0 writes the whole document on one line. The cases that
read text back parse it with `Config::parse_json`, and a real is compared bit for bit.

### WritesEveryKindOnOneLineInTheOrderOfTheKeys

At depth 0 an object of nine keys is written on one line that ends with a line feed. The keys
keep the order they were stated in, `zebra` before `apple`, an empty object is `{}` and an
empty array is `[]`.

### ExpandsTheFirstTwoLevelsAndKeepsDeeperOnesOnOneLine

At the default depth the root object and the values one level below it are expanded with four
spaces of indent per level. An array of numbers at the second level, each object inside
`layers` and an empty array stay on one line. The whole text is compared.

### EndsEveryLineWithALineFeedAndTheTextWithOne

The text ends with one line feed, the character before it is not a line feed, and the text
holds no carriage return.

### WritesARootScalarAlone

`Value::integer(-3)` is written as "-3" and a line feed.

### SpellsNumbersInTheShortestFormThatReadsBack

`[7, -3, 0, 0.001, 1e-3, 2.5, 1.0, 1e300]` is written as
`[7, -3, 0, 0.001, 0.001, 2.5, 1.0, 1e+300]`.

### AnIntegerReadsBackAsTheSameInteger

Eight integers, among them 9007199254740993 and the largest and the smallest int64, each read
back as `Kind::Int` with the same value.

### ARealReadsBackAsTheSameBits

Nineteen reals, among them -0.0, 0.1 + 0.2, 1e22, the largest and the lowest double, the
smallest normal, the smallest denormal and epsilon, each read back as `Kind::Real` with the
same bits.

### EscapesQuotesBackslashesAndControlCharacters

A string with a quote, a backslash, a solidus, the five short control escapes, the bytes 0x01,
0x1F, 0x7F and a NUL is written with `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u0001`,
`\u001f` and `\u0000`. The solidus and 0x7F are written as they are. The text reads back to the
same string.

### WritesUTF8AsItsOwnBytes

A string holding U+011F, U+65E5 and U+1F363 is written between quotes as its own bytes and
reads back to the same string.

### EscapesAKeyAsItEscapesAString

An object whose key holds quotes and a tab is written with `\"` and `\t` in the key, and its
keys read back equal.

### RefusesANumberThatIsNotFinite

A NaN fails with `InvalidDescription` and "not a number". Plus and minus infinity each fail
with "infinite".

### RefusesAStringThatIsNotUTF8

Six byte sequences fail with `InvalidDescription` and "UTF-8" as a string, and with "UTF-8" and
"key 0" as the key of an object: 0xFF, a lone continuation byte, an overlong solidus, a
three-byte form cut short, the surrogate U+D800 and U+110000.

### RefusesAnObjectThatStatesAKeyTwice

An object with the keys `rate`, `units`, `rate` fails with `InvalidDescription`, `'rate'` and
"twice".

### ARefusalNamesTheWayToTheValue

A NaN under `rate` in entry 1 of the array under `layers` fails with a message containing
"key 'layers'", "entry 1", "key 'rate'" and "not a number".

## test_model_writer.cpp

Suites: `Unit_ModelWriter`.

Tests `Config::write_model`. Three models are parsed from text in the file. The chain model has
three rows and states neither `from` nor `outputs`. The full model has a `defaults` block, an
attribute of every JSON kind, every form of a `from` entry and every form of an `outputs`
entry. The task model has a `task` with a value of every JSON kind. A round trip parses the
written text and compares the input shape and type, each layer's type, attributes and sources,
the outputs and the task, with a real compared bit for bit. The two example cases locate the
repository through `SA_TEST_FIXTURE_DIR` and report themselves skipped when it is not defined.

### WritesTheInputElementTypeOnlyWhenItIsInt32

The text of a float32 model does not contain "dtype". With `input_dtype` set to int32 the text
contains `"dtype": "int32"` and reads back as int32. With float64 the write fails with
`InvalidDescription` and "float64".

### RoundTripsTheMNISTExample

The model loaded from `applications/examples/mnist_cnn/model.json` round-trips.

### RoundTripsTheYOLO26Example

The model loaded from `applications/examples/yolo26/model.json` round-trips.

### RoundTripsEveryAttributeKindAndEveryFormOfFromAndOutputs

The full model round-trips.

### WritesTheSameTextFromWhatItWrote

The text written from the full model, parsed and written again, equals the first text.

### LeavesOutTheFromAndTheOutputsTheParserFillsIn

The chain model is written as `format_version`, `input` and three rows that hold a `type` and
their attributes, with no `from` and no `outputs`. The whole text is compared.

### WritesTypeThenFromThenTheAttributesAndTheOutputsLast

A model whose rows state `type` after their attributes is written with `type` first, then
`from`, then the attributes, and with `outputs` after `layers`. The whole text is compared.

### WritesADefaultIntoEveryRowAndNoDefaultsBlock

The text of the full model has no `defaults` key. Its first `conv` row carries `bn_epsilon`
0.001 and `bn_momentum` 0.03 after its own keys, and the `leaky_relu` row carries `slope`
0.001.

### RoundTripsAPortOfTheModelInputAtRowZero

The chain model whose row 0 reads the port `image` of `K_MODEL_INPUT` round-trips.

### AnEmptyOutputsListReadsBackAsTheLastRowPublishedWhole

With `outputs` cleared the text has no `outputs` key and reads back with one output: node 2,
with an empty port and an empty name.

### RefusesAModelWithoutLayers

A model with no layers fails with `InvalidDescription` and "no layers".

### RefusesAnInputShapeAModelFileCannotState

An empty input shape fails with "input shape" and "no extents". A shape whose first extent is
the largest int64 and whose second is 2 fails with "input shape" and "extent 1".

### RefusesAttributesThatAreNotAnObject

Null attributes on row 1 fail with `InvalidDescription`, "layer 1", `'relu'` and "null".

### RefusesAnAttributeNamedAsARowKey

An attribute named `type` on row 1 fails with "layer 1" and `'type'`. An attribute named `from`
on row 2 fails with "layer 2" and `'from'`.

### RefusesARowThatReadsNothingOrARowNotAboveIt

Row 1 with no source fails with "layer 1" and "reads nothing". Row 1 reading row 1, row 0
reading row 2 and row 2 reading row -2 each fail with the layer, the row and "not above".

### RefusesAPortOfTheModelInputBelowRowZero

Row 2 reading the port `image` of `K_MODEL_INPUT` fails with `InvalidDescription`, "layer 2",
`'image'` and "row 0".

### RefusesAnOutputAModelFileCannotState

An output at row 3 of three fails with "output 0", "row 3" and "0 to 2". A second output at
row -1 fails with "output 1" and "row -1". A named output without a port fails with "output
0", `'predictions'` and "without a port".

### ForwardsTheRefusalOfAnAttributeJSONCannotSpell

An infinite `rate` on row 2 fails with `InvalidDescription` and a message containing "key
'layers'", "entry 2", "key 'rate'" and "infinite".

### RoundTripsATaskOfEveryValueKind

The task model, whose `task` is an object, round-trips.

### WritesTheTaskAfterTheOutputsWithItsKeysInOrder

In the text of the task model the `outputs` key comes before the `task` key. The task reads
back with the keys `type`, `classes`, `anchors`, `heads`, `suppression`, in that order.

### LeavesOutANullTaskAndReadsItBackNull

The chain model has a null task, its text has no `task` key, and the task reads back null.

### RefusesATaskThatIsNotAnObjectWithAStringType

Four tasks fail with `InvalidDescription` and "task": an array ("an array", "object"), an
object without `type` ("no 'type'"), a `type` of 3 ("'type'", "a whole number") and a `type`
of "" ("'type'", "empty string").

## test_section_reader.cpp

Suites: `Unit_SectionReader`.

Tests `Config::SectionReader` and the free readers of
`modules/core/config/include/SushiAI/config/section_reader.hpp`. The reader of every case is
named "schedule" and the accepted keys are `type`, `base_rate` and `steps`.

### NamesAMemberAnEntryAndAList

`member_of("schedule", "steps")` is "schedule 'steps'", `entry_of` of that with 2 is "schedule
'steps' entry 2", and `list_quoted` of the three keys is "'type', 'base_rate', 'steps'" and of
an empty span "". `SectionReader::member("type")` is "schedule 'type'".

### CheckObjectRefusesAnotherKind

`check_object` is ok for `{}`. For a string it fails with `InvalidDescription` and "schedule
must be an object, but it is a string".

### CheckKeysNamesTheKeyAndListsWhatIsAccepted

`check_keys` is ok for a section with `steps` and `type`. For a section with `gamma` it fails
with "schedule states the key 'gamma', which it does not have; it accepts 'type', 'base_rate',
'steps'".

### CheckTypeAcceptsAnAbsentKeyAndRefusesAnotherType

`check_type("type", "step", "a step schedule")` is ok when `type` is absent and when it is
"step". For "cosine" it fails with "schedule 'type' is 'cosine'; a step schedule states
'step'", and for 3 with "schedule 'type' must be a string, but it is a whole number".

### FindReturnsTheMemberOrNamesTheAbsentKey

Over `{"base_rate": 0.5}`, `has` is true for `base_rate` and false for `steps`. `find` returns
a `Kind::Real` member for `base_rate` and fails for `steps` with "schedule has no 'steps'".

### FindOfChecksTheKind

Over `{"steps": 3}`, `find_of("steps", Kind::Int)` is ok. Asked for `Kind::Array` it fails with
"schedule 'steps' must be an array, but it is a whole number", and for the absent `type` with
"schedule has no 'type'".

### TheNumberReadersCheckTheKind

`read_real` returns 7.0 for the whole number 7 and 0.25 for the real 0.25, and `read_whole`
returns 7 for 7. `read_real` of a string fails with "rate must be a number, but it is a
string", and `read_whole` of 0.25 with "count must be a whole number, but it is a number".

### ReadFloat32RefusesWhatFloat32CannotHold

`read_float32` returns 0.5f for 0.5. For 1e39 it fails with `InvalidDescription`, "scale is"
and "which float32 does not hold as a finite value", and for a string with "scale must be a
number, but it is a string".

### TheStringAndArrayReadersCheckTheKind

`read_string` returns "step" for a string and `read_array` returns three entries for
`[1, 2, 3]`. `read_string` of an array fails with "type must be a string, but it is an array",
and `read_array` of a string with "steps must be an array, but it is a string".

## test_train_config.cpp

Suites: `Unit_TrainConfig`.

Tests `Config::parse_train`, `Config::check_start` and `Config::read_run_seed`. Most cases
parse a file that states `sushiai_train`, `model`, `objective`, `data` and an `sgd` optimizer,
followed by the members the case adds. The last four cases parse a file whose `data` block the
case supplies.

### AFileWithoutAScheduleHoldsTheNullValue

Without a `schedule` member, `spec.schedule` is null, the optimizer type is "sgd" and `epochs`
is 1.

### TheScheduleIsCarriedAsTheFileStatesIt

A `schedule` object with `type`, `base_rate` and a key named `unread_here` is kept as an object
of three keys. Its `type` is "step", its `base_rate` is a real and it has `unread_here`.

### AScheduleOfAnotherKindIsCarriedUnchecked

A `schedule` that is the string "later" parses, and `spec.schedule` has the kind `String`.

### AKeyBesideTheScheduleIsStillRefused

A member named `schedules` raises an error naming `'schedules'`.

### AFileWithoutWeightsHoldsAnEmptyPath

Without a `weights` member, `spec.weights` is empty.

### TheWeightsPathResolvesAgainstTheFile

Parsed with the directory `runs`, `"weights": "start.sushi"` gives `runs/start.sushi` and the
model path is `runs/model.json`.

### WeightsThatAreNoStringAreRefused

`"weights": 3` raises an error naming `'weights'`.

### TheResumePathResolvesAgainstTheFile

Without a `resume` member the path is empty. Parsed with the directory `runs`,
`"resume": "last.sushi"` gives `runs/last.sushi` and leaves `weights` empty.

### AResumeThatIsNoStringIsRefused

`"resume": true` raises an error naming `'resume'`.

### WeightsAndResumeTogetherAreRefused

`check_start` is ok for a file with neither member, with `weights` alone and with `resume`
alone. With both it fails with `InvalidDescription` and a message naming `'train.json'`,
`'weights'` and `'resume'`.

### TheSeedIsZeroUnlessTheFileStatesOne

Without a `seed` member `spec.seed` is 0 and `read_run_seed` returns 0. With
`"seed": 20261011` both are 20261011.

### ASeedThatIsNoIntegerIsRefused

The seeds "seven" and 0.5 each raise an error naming `'seed'`.

### ANegativeSeedIsRefused

A file with `"seed": -3` parses, and `read_run_seed` fails with `InvalidDescription` and a
message containing `'train.json'`, `'seed'` and "-3".

### ADataBlockWithAReaderTakesTheReaderType

A `data` block with `reader`, `task` and `root` and no `type` gets the type
`Config::K_READER_DATA_TYPE`. Its attributes keep all three keys and `reader` is "yolo_txt".

### ATypedDataBlockKeepsItsType

A `data` block with `"type": "mnist"` and `directory` gets the type "mnist". Its attributes
hold `directory` and no `type`.

### ATypeBesideAReaderStaysATypedBlock

A `data` block with `"type": "mnist"` and `reader` gets the type "mnist" and its attributes
still hold `reader`.

### ADataBlockWithNeitherKeyIsRefused

A `data` block with only `root` raises an error naming `'type'`.
