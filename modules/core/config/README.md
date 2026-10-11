# config

## Owns

The configuration documents. `Value` is the dynamically typed value that a JSON document parses
into, and `parse_json` and `parse_json_file` build it. The model document parses into
`ModelSpec` and its `LayerSpec` rows, and the training and data documents parse into `TrainSpec`
and `DataSpec`. `ModelSpec::input_dtype` holds the optional `dtype` of the `input` object:
`float32` when the key is absent, or `int32` for a model whose first row reads indices; any
other text is refused with the two names. `ModelSpec::task` holds the `task` object of a model file as a `Value`: the
module checks that it is an object with a string `type` and reads nothing else in it, so it
names no task type. `TrainSpec::schedule` holds the `schedule` object of a training file the same
way, null when the file has none; this module checks nothing in it, and `Optim::read_rate_schedule`
reads it. `TrainSpec::weights` holds the `weights` path of a training file, resolved against the
file's directory as `model` and `checkpoint` are, and empty when the file has none.
`TrainSpec::resume` holds the `resume` path the same way: the checkpoint whose model state and
run state a run goes on from. `check_start` returns `InvalidDescription` for a file that states
both. `TrainSpec::seed` holds the optional integer `seed` of a training file, 0 when the file
has none; a value of another kind is refused by `parse_train` with the key. `read_run_seed`
returns it as the `uint64_t` a training graph takes, or `InvalidDescription` with the file and
the value when it is negative. The range check sits there and not in `parse_train` because
`parse_train` reports by throwing and a new failure site returns a status. A `data` block
that spells `reader` and no `type` parses into a `DataSpec` whose type is `K_READER_DATA_TYPE` and
whose attributes are the whole block; any other block needs a `type`, as before.
`write_json` turns a `Value` back into JSON text and `write_model` turns a `ModelSpec`
back into the text of a model file. Each document has a version constant: `K_MODEL_VERSION`, `K_TRAIN_VERSION` or
`K_DATA_VERSION`. `find_entry` looks up a type name in a static registry table, for the modules
that select a row from a file.

## Depends on

The module declares `core` as a public dependency. Its JSON parser comes from `nlohmann_json`,
which the module takes through `EXTERNAL_PRIVATE_INCLUDES`: the module receives the include
directories and definitions of that target, and the target name stays out of the module's link
interface. It declares no other private dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/config/value.hpp` | `Kind`, `kind_name`, `Value` with its static factories `boolean`, `integer`, `real`, `string`, `array` and `object`, and its accessors `as_bool`, `as_int`, `as_real`, `as_string`, `as_array`, `has`, `at`, `at_or`, `reject_unknown_keys`, `without`, `require_version`, `parse_json`, `parse_json_file` |
| `SushiAI/config/json_writer.hpp` | `K_JSON_EXPANDED_DEPTH`, `write_json` |
| `SushiAI/config/model_config.hpp` | `K_MODEL_VERSION`, `K_MODEL_INPUT`, `SourceSpec`, `OutputSpec`, `LayerSpec`, `ModelSpec`, `parse_model`, `load_model` |
| `SushiAI/config/model_writer.hpp` | `write_model` |
| `SushiAI/config/train_config.hpp` | `K_TRAIN_VERSION`, `K_DATA_VERSION`, `K_READER_DATA_TYPE`, `TrainSpec`, `DataSpec`, `parse_train`, `load_train`, `check_start`, `parse_data`, `load_data` |
| `SushiAI/config/json_events.hpp` | `K_JSON_EVENT_BUFFER_BYTES`, `JSONEvents`, `read_json_events` |
| `SushiAI/config/registry.hpp` | `find_entry` |
| `SushiAI/config/section_reader.hpp` | `KeySpan`, `member_of`, `entry_of`, `list_quoted`, `read_real`, `read_whole`, `read_float32`, `read_string`, `read_array`, `SectionReader` |

## Reading a document as events

`parse_json_file` holds the text, the parser's document and the `Value` tree at once, which
is right for a configuration file and too much for a file of hundreds of megabytes.
`read_json_events(path, handler)` builds none of the three. It streams the file through a
buffer of `K_JSON_EVENT_BUFFER_BYTES`, 64 KiB, and calls one method of the `JSONEvents` handler
per token: `begin_object`, `end_object`, `begin_array`, `end_array`, `key`, `null`, `boolean`,
`whole`, `real` and `string`. A key and a string are views that last for the call. A whole
number above the largest `std::int64_t` arrives as `real`.

Every event returns whether the read goes on. An event that returns false ends the read, and
`read_json_events` returns success: the handler chose to stop and keeps its own reason.

| Status | When |
| --- | --- |
| `FileNotFound` | The file cannot be opened |
| `InvalidFile` | The text is no JSON document; the message is `'<path>' is not valid JSON at byte <N>: <the parser's text>`, where N counts the bytes read when the error was met, the end of the input included |
| `InvalidFile` | A read fails |
| `OutOfMemory` and the other codes of `Detail::guard` | The handler throws |

The parser is the SAX interface of `nlohmann_json`, 3.12.0 as installed; its types stay in
`source/json_events.cpp`. `Vision::Datasets::read_coco_json` reads through it.
`tests/test_json_events.cpp` covers the order of the events, a document longer than the buffer,
a handler that stops and each refusal.

## Reading a section without throwing

`Value::as_int` and its siblings throw. A module whose reader returns a `Status` checks the
kind first, and `section_reader.hpp` holds those checks once. `SectionReader(section, where)`
binds an object and the name a refusal gives it; `member_of` and `entry_of` build the names of
a key and of a list entry from it.

| Call | Refusal, `InvalidDescription` |
| --- | --- |
| `check_object()` | `<where> must be an object, but it is <kind>` |
| `check_keys(allowed)` | `<where> states the key '<key>', which it does not have; it accepts <list>` |
| `check_type(key, expected, what)` | `<where> '<key>' is '<stated>'; <what> states '<expected>'`; an absent key passes |
| `find(key)` | `<where> has no '<key>'` |
| `find_of(key, kind)` | `<where> '<key>' must be <kind>, but it is <kind found>` |
| `read_real`, `read_whole`, `read_string`, `read_array` | `<where> must be a number`, `a whole number`, `a string`, `an array`, `but it is <kind>` |
| `read_float32` | As `read_real`, or `<where> is <value>, which float32 does not hold as a finite value` |

The schedule sections of `core/optim`, the anchor task and the head training section of
`vision/detect`, the `sum` objective of `layers/train` and the sections of
`language/generation` and `language/datasets` read through it.
`tests/test_section_reader.cpp` pins every text above.

## What the writers write

`write_json` keeps the order of the keys. The first `K_JSON_EXPANDED_DEPTH` levels take one line
per member with four spaces of indent, and a deeper container takes one line. An integer is
written in decimal digits. A real is written in the shortest form that parses back to the same
double, with `.0` added when that form has neither a point nor an exponent. Text is UTF-8 with
line feed line ends. It refuses a number that is not finite, a string or key that is not UTF-8, and an
object that states a key twice.

`write_model` writes `format_version`, `input`, `layers`, `outputs` and `task`, in that order,
and each row as `type`, `from`, then its attributes. `input` carries `dtype` only for an
`int32` input, and a specification of another element type is refused. It writes `task` as the specification holds
it and leaves it out when it is null. It writes no `defaults` block: every attribute is in
its row, because a `ModelSpec` does not record which keys a default supplied. It leaves out a
`from` that reads the row before and an `outputs` that publishes the last row whole, since the
parser fills both in. It refuses a specification that no model file can state.
