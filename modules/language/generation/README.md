# generation

## Owns

What turns the logits of a decoder into a continuation of a prompt: the `Sampler` seam and its
four samplers, the reader of a sampler section, the reader of the `next_token_generation` task
section, and the `Generator` seam with its two loops. `WindowGenerator` runs the whole window
for each new token. `CachedGenerator` runs one position per token over a step that carries the
keys and the values of the positions before it. Everything is in namespace
`SushiAI::Language::Generation`.

## Depends on

- `failure`, `dtype`, `shape`, `seam`, `config`, `data` and `train` (`PUBLIC_DEPENDS`): `Status` and
  `Result`,
  `Config::Value` for the two sections, `Data::Batch` for the window a run feeds, and
  `Train::EvaluationStep`. Of the core modules, `dtype`, `shape` and `seam` are private
  (`PRIVATE_DEPENDS`).
- `language/text` (`PUBLIC_DEPENDS`), for `Text::TokenID`.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/generation/sampler.hpp>` | `Sampler`, `RandomGenerator`, `LogitSpan` |
| `<SushiAI/language/generation/greedy_sampler.hpp>` | `GreedySampler`, `K_GREEDY_SAMPLER` |
| `<SushiAI/language/generation/temperature_sampler.hpp>` | `TemperatureSampler`, `K_TEMPERATURE_SAMPLER` |
| `<SushiAI/language/generation/top_k_sampler.hpp>` | `TopKSampler`, `K_TOP_K_SAMPLER` |
| `<SushiAI/language/generation/nucleus_sampler.hpp>` | `NucleusSampler`, `K_NUCLEUS_SAMPLER` |
| `<SushiAI/language/generation/sampler_section.hpp>` | `make_sampler`, `K_DEFAULT_TEMPERATURE` |
| `<SushiAI/language/generation/generation_task.hpp>` | `GenerationTask`, `read_generation_task`, `K_GENERATION_TASK_TYPE`, `K_DEFAULT_GENERATED_TOKENS`, `K_DEFAULT_GENERATION_CACHE` |
| `<SushiAI/language/generation/generator.hpp>` | `Generator` |
| `<SushiAI/language/generation/window_generator.hpp>` | `WindowGenerator` |
| `<SushiAI/language/generation/cached_generator.hpp>` | `CachedStepLeaves`, `RingStepLeaves`, `AngleRows`, `CachedGenerator` |
| `<SushiAI/language/generation/position_feed.hpp>` | `Detail::PositionFeed`, `Detail::make_position_feed` |
| `<SushiAI/language/generation/ring_position_feed.hpp>` | `Detail::RingPositionFeed` |
| `<SushiAI/language/generation/table_position_feed.hpp>` | `Detail::TablePositionFeed` |

## Samplers

A sampler reads the logits of one position, one `float` per token, and returns the index of
the token it picks. The caller owns the generator and seeds it. `RandomGenerator` is
`std::mt19937_64`, whose output the C++ standard fixes for a seed, and a draw is its upper 53
bits scaled into `[0, 1)`, so a seed gives the same tokens on every platform. No sampler uses
a distribution of the standard library, whose results differ between implementations.

| Sampler | Section | Picks | Draws |
| --- | --- | --- | --- |
| `GreedySampler` | `{ "type": "greedy" }` | The largest logit; the lowest index among equal ones | 0 |
| `TemperatureSampler` | `{ "type": "temperature", "temperature": 0.8 }` | A token with probability `softmax(logits / temperature)` | 1 |
| `TopKSampler` | `{ "type": "top_k", "k": 40, "temperature": 0.8 }` | The same, over the `k` largest logits only | 1 |
| `NucleusSampler` | `{ "type": "nucleus", "p": 0.9, "temperature": 0.8 }` | The same, over the fewest most probable tokens whose probabilities reach `p` | 1 |

`temperature` is 1 when a section leaves it out; it is positive and finite. `k` is at least 1,
and a row with fewer logits keeps all of them. `p` is above 0 and at most 1. Where equal logits
meet the edge of what top-k or nucleus keeps, the lower indices are kept. A key the type does
not read is refused, and an unknown type lists the four.

The three samplers that draw share `source/weights.hpp`: the weights
`exp((logit - largest) / temperature)`, their sum, and a draw by running sum over them. A
sampler allocates one vector of doubles per call, or two for top-k and nucleus. A logit that
is not finite is the caller's to refuse, and both generators do so before they sample.

## The task section

A model file whose `task.type` is `next_token_generation` is read by `read_generation_task`.

```json
"task": {
    "type": "next_token_generation",
    "vocabulary": "vocab.savb",
    "window": 256,
    "tokens": 200,
    "seed": 1,
    "cache": true,
    "sampler": { "type": "top_k", "k": 40, "temperature": 0.8 }
}
```

| Key | Required | Holds |
| --- | --- | --- |
| `vocabulary` | Yes | The path of the vocabulary file the ids decode with, as the file spells it |
| `window` | Yes | `T`, the positions of the window; at least 1 |
| `tokens` | No | The tokens to generate; 64 when absent |
| `seed` | No | The seed of the generator; 0 when absent |
| `sampler` | No | A sampler section; greedy when absent |
| `cache` | No | `true` for `CachedGenerator`, `false` for `WindowGenerator`; `true` when absent |

The reader resolves no path and opens no file. `sushiai inference` resolves `vocabulary`
against the directory of the model file and refuses a `window` that is not the model's input
shape.

## WindowGenerator

`Generator` is the seam of both loops: `window()`, `vocabulary()` and `generate(prompt, count,
sampler, random, generated)`. The checks of a prompt, a count and a row of logits are shared in
`source/request.hpp`, so both loops refuse the same things in the same words.

`WindowGenerator::make` binds a `Train::EvaluationStep` traced for `[1, T]` int32 ids and one
`[1, T, V]` float32 output. `generate` keeps the last `T` ids of the prompt as the context.
For each new token it writes the context to the front of the window and zeros behind it, runs
the step, samples from the logits at the last position the context fills, and appends the
token. A context of `T` ids drops its first id before it takes the new one.

The loop is right for a model whose position `t` reads positions `0` to `t` only, which is what
a causal mask gives: the zeros behind the context reach no position that is sampled. It costs
one forward pass of the whole window per token, so `n` tokens cost `n` passes of `T` positions
whatever the length of the context. It is the reference a cached path has to equal.

## CachedGenerator

`CachedGenerator::make` binds a `Train::EvaluationStep` that carries state and publishes one
`[1, 1, V]` float32 output. Every run feeds it the model input, `[1, 1]` int32, the id of one
token, and a `[1, T]` float32 mask. The leaves that place the token are one of two sets:

| The step is fed | Feed | A slide of the window costs |
| --- | --- | --- |
| A `[1, 1]` int32 position | `TablePositionFeed` | A reset and `T` runs |
| A `[1, 1]` int32 cache slot and `[1, n]` float32 cosines and sines | `RingPositionFeed` | One run |

The names of the leaves, the masked score and the function that fills the angles arrive in a
`CachedStepLeaves`, so this module names no block. `sushiai inference` passes the names of
`language/blocks/incremental.hpp` and `Blocks::write_rotary_row`. `make` picks the feed from
the leaves the step is fed and reads no model file: a step fed both the position and the
slot, or neither, is refused, and so is a step fed a leaf beyond the id, the mask and those of
its feed. Such a step comes from `Train::build_incremental_graph` over an `NN::ModelGraph` of
the language rows, whose blocks declare a position under a learned table and a slot under
rotary positions.

`generate` keeps the last `T` ids of the prompt as the context, calls `reset()` on the step,
which zeroes the keys and the values of every block, and runs the step once per id of the
context. The mask of a run is 0 at each cache row a fed token fills and the masked score at
the others. It then samples from the logits of the last run, runs the step for the new token
at the next position, and so on. The token a run ends on is not fed, since nothing reads its
logits. A context of `T` ids drops its first id before it takes the new one, as
`WindowGenerator` does.

| Generator | Context | Runs per token | Positions per run |
| --- | --- | --- | --- |
| `CachedGenerator`, either feed | Shorter than `T` | 1 | 1 |
| `CachedGenerator`, position | `T`, so the window slides | `T` | 1 |
| `CachedGenerator`, cache slot | `T`, so the window slides | 1 | 1 |
| `WindowGenerator` | Any length | 1 | `T` |

The two feeds are `table_position_feed.hpp` and `ring_position_feed.hpp` behind
`Detail::PositionFeed` of `position_feed.hpp`: `place(position)` writes the
leaves of one token, `feeds()` lists them for the run, and `keeps_rows_on_slide()` is the one
thing the loop asks before it decides between a run and a restart.

### A step fed a position

A learned position table gives every kept id a new position when the window slides, so no
cached row stays right: the generator resets the step and feeds the `T` ids again. Each token
past the window therefore costs `T` runs of one position, which is no less work than one pass
of the window and pays the overhead of a run `T` times. The cache saves work only until the
context reaches `T`. A prompt of 3 ids continued by 12 tokens over a window of 8 takes 56
runs, 48 of them in the six slides.

Under a greedy sampler the tokens equal those of `WindowGenerator` for a prompt that fits and
for one whose context slides; `tests/language/generation/test_cached_generator.cpp` checks both, the
logits of one position against the window's, and that the step is compiled once.

### A step fed a cache slot

The run of the token at position `p`, counted from the reset, is fed the slot `p mod T` and
the cosines and the sines of `p`. Position `T` writes over the row of position 0, which holds
the token the window drops, and every other row stays as it is: a rotary key is turned by the
position of its own token, and the score of a query with a key depends on the difference of
their positions alone (README of `language/blocks`, "The ring of a rotary attention"). A
slide is therefore one run, as any other token is, and the mask is all zeros from position
`T - 1` on. The 12 tokens above take 14 runs: 3 for the prompt and 11 for the tokens that are
fed.

`AngleRows` computes the angle in double precision. Its rounding error stays below the
resolution of a float32 cosine while `p` is below `2^28`, about 268 million tokens since the
last reset. Nothing refuses a position past that.

### What a ring holds after a slide

The keys and the values of the first block are functions of one token, so a model of one
block gives the logits of `WindowGenerator` at every position, before and after a slide.

From the second block on, the row of a token was computed from what the token could read when
it was fed: the `T` tokens that end at it. `WindowGenerator` computes every position again
over the present window, where the token at window position `i` reads the `i` tokens before
it and nothing older. After the first slide a ring of two or more blocks therefore holds rows
that read further back than the window form does, and its logits differ from those of
`WindowGenerator` by more than rounding. Until the first slide the two agree at any depth.

What the ring computes is the model over a cache of unbounded rows in which each position
reads the last `T`: `tests/language/generation/test_cached_generator.cpp` checks a two-block ring
against a 64-row cache that never wraps and is masked to the last 8 positions, token for
token over 26 tokens. The same file checks a one-block ring against `WindowGenerator` over 26
tokens and its logits at each of 30 fed positions, the count of runs, and that a slot written
twice holds the second row. A model that must equal the window form at depth after a slide
runs with `"cache": false`.

### What the layers below give it

| Need | Seam |
| --- | --- |
| Three or five leaves fed per run | `NN::Trace::feed` declares a named leaf in a carrying trace; `Train::TracedStep::fed` lists them; `EvaluationStep::run_by_name` and `run_in_order` feed them |
| State written by a forward-only run | `NN::Trace::carried_leaf` and `carry` declare a value the step keeps; `EvaluationStep` records `leaf = next` behind the graph with `Train::BufferUpdateRecorder`, and `reset()` fills the leaves |
| The write of one row | `scatter_add_rows` of the row into a zero `[T, dim]`, added to the cache under a position, or put in place of the row through `select` under a cache slot; see the README of `language/blocks` for its cost |
| A second trace form of a block | `NN::Node::forward_incremental`, `ModelGraph::trace_incremental` and the `forward_incremental` of each block |

The keys and the values are leaves of the step and not `NN::Buffer` state of a block.
`IO::load_checkpoint` requires every entry of a `NN::ModelState` to be supplied, so a buffer
listed in a block would make every checkpoint written before it unreadable, and a cache is
what one run holds and no part of a model.

Source: `docs/design/LANGUAGE/README.md`, phase 5.
