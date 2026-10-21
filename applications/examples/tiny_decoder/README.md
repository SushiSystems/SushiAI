# tiny_decoder: a decoder-only transformer of 10.5 million parameters, from JSON

Three JSON files describe a five-block decoder over byte-pair tokens and the runs that train
it and continue a prompt with it. There is no C++ in this directory.
`applications/programs/source/train.cpp` and `applications/programs/source/inference.cpp` read
the files, as they do for [`mnist_cnn`](../mnist_cnn/README.md), and
`applications/programs/source/tokenize.cpp` prepares the data.

No run has been made with these files, and the programs have not been built since the rows
they name were written. The values are untuned first guesses.

## The data

The repository ships no corpus. Put a text file at `data/tiny_decoder/corpus.txt` under the
repository root, then learn a vocabulary of 4096 tokens from it and encode it:

```bash
sa tokenize train  data/tiny_decoder/corpus.txt --vocabulary-size 4096 \
                   --output applications/examples/tiny_decoder/vocab.savb
sa tokenize encode data/tiny_decoder/corpus.txt \
                   --vocabulary applications/examples/tiny_decoder/vocab.savb \
                   --output data/tiny_decoder/corpus.satk
```

`train` holds the whole corpus in memory, and `encode` reads it one mebibyte at a time and
holds four bytes for each token it has written. A corpus too small to have 3839 pairs that
occur twice gives a vocabulary below the model's 4096 rows. Such a file trains, since the
objective refuses only a token file whose vocabulary is above the head's, and the rows no id
names stay as they were initialised.

`sa` starts every program from the repository root, so `data/tiny_decoder/corpus.satk` in
`train.json` and the `vocabulary` path in `eval.json` are relative to it. The `vocabulary` of
the `task` section in `model.json` is relative to `model.json`, which is why the vocabulary
file sits in this directory.

## Running it

```bash
sa build
sa train      applications/examples/tiny_decoder/train.json
sa inference  applications/examples/tiny_decoder/model.json \
              applications/examples/tiny_decoder/tiny_decoder.sushi \
              applications/examples/tiny_decoder/eval.json
```

`sa train` prints the loss and the perplexity of the objective. `sa inference` prints the
prompt of `eval.json` followed by 200 generated tokens, sampled among the 40 most probable at
a temperature of 0.8 under seed 1. Each token costs one forward pass of the whole window; the
module `modules/language/generation` has no key and value cache.

This file quotes no loss, perplexity or timing, because none has been measured.

## The files

| File | Describes |
| --- | --- |
| `model.json` | The architecture and the generation task |
| `train.json` | The objective, the data, the optimizer, the epochs, the checkpoint |
| `eval.json` | The prompt to continue |

`model.json` states `"dtype": "int32"` in `input`, since its first row reads token ids. The
shapes below leave out the batch axis.

| Row | Type | Output shape |
| --- | --- | --- |
| input | | `[256]` int32 |
| 0 | `embedding`, 4096 rows of 384 | `[256, 384]` |
| 1 | `position_embedding`, 256 rows | `[256, 384]` |
| 2 to 6 | `transformer_block`, 6 heads, feed-forward width 1536 | `[256, 384]` |
| 7 | `language_head`, tied to row 0 | `[256, 4096]` |

## The parameter count

10 544 256, counted by hand with `V = 4096`, `T = 256`, `D = 384`, a feed-forward width of
`4 D` and 5 blocks. `tests/language/registration/test_registration.cpp` checks the same formula
against `NN::parameter_elements` on a model of `D = 16`; no run has counted this model.

| Part | Elements | Formula |
| --- | --- | --- |
| Token table | 1 572 864 | `V D` |
| Position table | 98 304 | `T D` |
| One block | 1 774 464 | `12 D^2 + 13 D`: two normalisations of `2 D`, `qkv` of `3 D^2 + 3 D`, `proj` of `D^2 + D`, a feed-forward of `8 D^2 + 5 D` |
| Five blocks | 8 872 320 | |
| Head | 768 | `2 D`, its normalisation; the tied head has no weight of its own |
| Total | 10 544 256 | |

## Limits

- The head scores with the token table, so the table is counted once and has one gradient.
- `"dropout"` rows are left out: a compiled step replays one recorded mask, so a model that
  states a rate trains without dropout today. See `docs/design/LANGUAGE/README.md`.
- The training file states no `schedule`.
  [`docs/guides/CLI_GUIDE.md`](../../../docs/guides/CLI_GUIDE.md) documents the key.
- One batch holds 32 windows of 256 ids, and its logits are `32 * 256 * 4096` floats, which
  is 128 MiB before the gradients.
- An unknown key in any block is an error.
