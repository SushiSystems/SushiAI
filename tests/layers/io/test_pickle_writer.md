# tests/layers/io/test_pickle_writer.cpp

Suites: `Small_PickleWriter`.

Tests `IO::write_pickle`, in `modules/layers/io/include/SushiAI/io/pickle_writer.hpp`, against `IO::read_pickle`. The reader is the oracle: a tree that goes in must come out.

## TheReaderReadsBackEveryKind

A tree of every node kind the writer takes: none, booleans, integers of several sizes, a real, short and long bytes, lists, tuples, dictionaries, a persistent id and a call with appended items and built-on state. The reader gives back each node's kind, value and children.

## ASetIsRefused

A set fails with `Unsupported`, naming its path `root[0]`, because protocol 2 cannot state one.
