# scratch_files

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

`ScratchFile`, a named file under the temporary directory. It removes a file a previous run
left under that name when it is made, and removes the file again when it goes.

## Depends on

Nothing beyond the standard library.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/scratch_files/scratch_file.hpp` | `ScratchFile` |
