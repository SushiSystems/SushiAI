# modules/vision/imaging/include/SushiAI/vision/imaging/random.hpp


## Random

SplitMix64. The state is one 64-bit counter. A draw adds `K_INCREMENT` to the counter modulo
2^64 and returns `mix` of the sum:

```
z = value
z = (z ^ (z >> 30)) * K_FIRST_MULTIPLIER      (mod 2^64)
z = (z ^ (z >> 27)) * K_SECOND_MULTIPLIER     (mod 2^64)
mix = z ^ (z >> 31)
```

| Draw | Mapping | Draws consumed |
| --- | --- | --- |
| `next_u64()` | the 64 bits | 1 |
| `next_u32()` | `next_u64() >> 32` | 1 |
| `uniform()` | `(next_u64() >> 40) * 2^-24`, exact in `float`, in [0, 1) | 1 |
| `uniform(low, high)` | `low + (high - low) * uniform()` in `float`; a result that rounds to `high` becomes `std::nextafter(high, low)` | 1 |
| `index(count)` | `(uint64(next_u32()) * count) >> 32` | 1 |
| `bernoulli(p)` | `uniform() < p` | 1 |

Every mapping consumes exactly one draw whatever its arguments. The number of draws one sample
makes is not constant: it depends on whether the epoch is closed and on whether the mosaic gate
opened (7, 11 or 13; see the table under `pipeline.hpp`). That does not couple samples, because
`TrainingPipeline::make_sample` builds a generator of its own for each sample and no caller
passes one in. `index` rejects nothing; the probability of a value differs from `1 / count` by
at most 2^-32.

`derive(key)` returns a generator whose counter starts at `mix(state ^ mix(key + K_INCREMENT))`
and leaves the parent untouched. `make_sample_random(seed, epoch, index)` in `pipeline.hpp` is
the one place that applies it: `Random(seed).derive(epoch).derive(index)`.

The reference draws from Python's `random` and from `np.random` (Mersenne Twister, seeded per
worker at `data/build.py:233-235`). This module does not reproduce those sequences. A run here
and a run of the reference see different augmentations drawn from the same distributions.
