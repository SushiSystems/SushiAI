# umbrella

## Owns

The umbrella module holds one header, `SushiAI.hpp`, which includes all 112 public headers of the
`core` and `layers` tiers and declares nothing itself. `modules/CMakeLists.txt` names this header
as the umbrella of the top-level `sushiai` library. The `vision` tier has its own umbrella and is
not included here.

## Depends on

From `CMakeLists.txt`: `PUBLIC_DEPENDS core version graph config autograd optim data eager nn io
train`. The header includes a header of each of these modules.

## Public entry points

| Header | Declares |
| --- | --- |
| `SushiAI.hpp` | Nothing; includes every public header of `core` and `layers` |
