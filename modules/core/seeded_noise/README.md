# seeded_noise

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

`HostNoise`, a 64-bit splitmix step mapped to (-1, 1). It is reproducible from its seed and
brings no device random stream into a test that checks something else. It is not meant as a good
generator.

## Depends on

`core`, for `SushiRuntime::span`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/seeded_noise/host_noise.hpp` | `HostNoise` |
