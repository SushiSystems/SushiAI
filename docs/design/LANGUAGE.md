# Language

**Status:** Open — phases 1 to 5 and 7 of 7 written, not yet built; phase 6 waits by the owner's ruling; ruled into 0.1.0 on 2026-10-11

How SushiAI trains and runs a decoder-only transformer on text, and an encoder that answers
typed questions in one forward pass: the second domain after vision, laid out as [`DOMAIN_MODULES.md`](DOMAIN_MODULES.md) lays out a domain.

## Rulings

Made by the owner on 2026-10-11.

| Subject | Ruling |
| --- | --- |
| Scope | A language domain is inside the feature freeze of 0.1.0 |
| First model | A decoder-only transformer of nanoGPT scale |
| Reference | A PyTorch model of that scale, run outside the repository to produce tensors. Nothing of its source enters the repository |
| Several machines | Training across the owner's five machines through SushiRuntime's distributed layer is wanted. It has its own design, not written: it waits on what the machines and the network between them are |
| Mixture of experts | Asked about. Phase 7 below; possible on one machine with the operations phases 1 and 2 add |
| Decision model | A non-autoregressive model that answers typed questions about a text, of the kind TypeSafe AI's Jev is; the owner will use it in his games. Phase 6 below |

## The tier

`language` is a domain tier beside `vision`: one word in the tier table, a library
`sushiai_language` (`SushiAI::language`), an umbrella `SushiAI/language.hpp`, a test
binary, and `SA_BUILD_LANGUAGE`. No module of it depends on `vision`, nor the reverse.

| Module of `modules/language/` | Holds |
| --- | --- |
| `text` | The token type, the byte-pair tokenizer (its trainer and its encoder are two units), the vocabulary file |
| `blocks` | Causal self-attention, the feed-forward block, the transformer block, the output head, built from `nn` layers |
| `datasets` | The reader of a tokenised file as fixed-length windows, and its collator into the roles of [`DATASETS.md`](DATASETS.md) |
| `training` | The next-token objective and its metrics (loss, perplexity) |
| `generation` | Sampling by temperature, top-k and nucleus, and the cache of keys and values |
| `registration`, `umbrella` | As in vision |

Layers that are not specific to language go to `nn`, as `Conv2d` did: `Embedding`, `LayerNorm` and
`Dropout`, beside the `Linear` that exists.

## What the core tier gains

A domain adds layers and algorithms, not operations. Each of these is an operation, a lowering
and a gradient rule in the core tier, in its operation's domain file.

| Operation | Needed for | Gradient |
| --- | --- | --- |
| Gather of rows by an integer index | The token embedding, and later the routing of experts | Scatter-add into the table |
| An integer input leaf | Token ids reach the graph as int32, where every input is floating today | None |
| Layer normalisation over the last axis | Every block | Its own rule, as batch normalisation has |
| A causal mask on attention scores | Attention that does not read the future | Passes through the kept entries |
| Cross entropy over `[B, T, V]` | The objective | Whether a reshape to rows over the existing loss is enough is not checked |
| Sine and cosine | Rotary positions, if chosen over a learned position table | Each other |

Shared with the other stages and written once: gradient clipping by global norm, parameter
groups (no decay on biases, normalisation gains and embeddings), the cosine schedule with
warm-up, and one weight bound to two places (the embedding table and the output head).

## Phases

| Phase | Work | Ends when |
| --- | --- | --- |
| 1 | Gather and scatter-add, the integer leaf, `Embedding` | A gradient check on an engine passes |
| 2 | Layer normalisation, the causal mask, `LayerNorm`, the attention and transformer blocks | One block's forward pass equals the reference's |
| 3 | The tokenizer, the vocabulary file, the windowed reader | A text encodes and decodes to itself; a window batch has the stated roles |
| 4 | The objective, clipping, parameter groups, the schedule, weight tying | The model's loss and one update equal the reference's, then it overfits one batch |
| 5 | Generation with the key and value cache | Greedy generation from imported weights equals the reference's tokens |
| 6 | The decision model below: the encoder, the typed heads, calibration | A model trained on a labelled set answers each question type, and its expected calibration error on a held-out split is under a stated bound |
| 7 | Mixture of experts: a top-k router, the dispatch and the gather of expert outputs, the balance term | A block with experts trains on one machine and its balance term keeps every expert in use |

The first training run is a model of 10 to 30 million parameters on a small corpus, chosen so
that an epoch takes hours on one RTX 3080 Ti with 12 GB.

## What phase 2 left open

- **`Dropout` draws its mask on the host.** A lowering row registers a prelude that
  `CompiledStep::run` calls before each replay, and the prelude of `K_DROPOUT` fills a
  FLOAT32 scratch from SplitMix64 keyed by site seed, run and element: about 12.6 MB per
  site per step for a `[32, 256, 384]` activation. `record_keep_draw` is the one function
  that changes when SushiBLAS carries a positioned Bernoulli fill (`GRAPH_CAPABILITIES.md`,
  B1). The `seed` key of the training file seeds the sites, and a resumed run restores the
  run count, so it draws the masks the uninterrupted run would have drawn.
- **GELU is the tanh approximation**, so the reference is run with the same one.
- **The masked score is -1e9**, which is minus infinity in half precision.

## What phase 3 decided and left open

- **Merges do not cross a run of blank bytes**, and a chunk is cut at 256 bytes. It bounds the
  trainer and the encoder; the alternative is one unbroken byte sequence.
- **Ids**: bytes are 0 to 255, special tokens follow with end-of-text at 256, merges come last
  in learned order. A special token decodes to no bytes and the encoder never emits one.
- **The files**: a vocabulary file (`SAVB`) and a token file (`SATK`), both versioned and
  little-endian, laid out byte by byte in the READMEs of `text` and `datasets`.
- **Windows are strided.** `Dataset::read` has no epoch argument, so starts drawn anew at each
  epoch have nowhere to live.
- **A dataset states its input's element type** (`Dataset::sample_dtype`), so `TokenDataset`
  publishes int32 and `Trainer::fit` refuses a dataset whose input differs from the leaf.
- **The trainer's time on tens of megabytes is not measured.**

## What phase 4 decided and left open

- **The loss reads integer labels through the row gather**, not a one-hot leaf, which would be
  256 MiB written from the host at each step for a batch of 32 windows of 256 over 8192
  tokens. The cost: the gradient goes through the scatter-add and the unfused log-softmax
  rule, and `N * T * V` must fit in int32.
- **`Trainer` runs it.** An objective fills its own leaves and reports its own metrics, so
  the loop names no task. What `sushiai train` still lacks is a dataset row that publishes
  token windows, and node rows for the blocks.
- **Weight decay reaches weights only.** The rule is in one place, by parameter role, and
  holds for every run through `bind_parameters`; `decay_all` restores decay on every slot.
- **The clip is recorded in the step**: one norm per gradient, one factor, and each update
  reads the factor, so no value returns to the host.
- **A schedule is a variant of the two kinds** a file may name, `step` and `cosine`.
- **A tied head holds the address of the embedding's parameter.** A model file states the tie
  as a row reference, and the table then lives in a node the model graph owns.
- **Every number the overfit test asserts is unverified**: 400 updates, a last loss under a
  tenth of the first.

## What phase 5 decided and left open

- **A decoder is reachable from files**: `input.dtype` in the model file, the `tie` of a
  `language_head` row, the block rows and the `tokens` dataset row of `language/registration`,
  and `sushiai tokenize` for the vocabulary and the token file.
- **A row that needs another row is a linked row**: its factory receives the rows above it.
  The other factories kept their signature.
- **The position table is added in one node**, by viewing the sequence as rows of `T * D`.
- **Samplers draw from a generator the caller owns**, and use no library distribution, so a
  seed gives the same tokens on every platform.
- **Generation runs the whole window for each token.** It is the reference: correct and slow.
- **The key and value cache is written** on three seams that name no language: a step fed by
  several named leaves, a step that carries values between runs, and an incremental form a
  node may implement. The keys and values are leaves of the step, not state of the model, so
  no checkpoint changes.
- **With the learned position table the cache saves work only until the context is full.**
  A slide resets it and feeds the whole window again, one position at a time. A row is
  written as a scatter-add into zeros, four passes over `[T, dim]` per cache per token; an
  in-place row write needs a graph operation that does not exist.
- **`"positions": "rotary"` keeps the cache across a slide.** The attention turns queries and
  keys by their absolute position, the cache is a ring of T rows written at the position
  modulo T, and a token costs one run: 1 966 080 multiply-adds per block at T = 256 and
  dim = 384 against 503 316 480. The cosine and sine tables are host tables of the step
  (`Trace::host_table`), not state, so no checkpoint holds them.
- **A ring of two or more blocks does not equal the window form after a slide.** A row above
  the first block was computed from the T tokens before its own, where the window form
  recomputes each position over the present window. One block is equal exactly; a deeper
  ring equals a cache that never wraps, masked to the last T positions. `"cache": false`
  generates through the window. A deep rotary model defaults to the ring, by the owner's
  ruling of 2026-10-11.
- **`sushiai inference` continues a prompt** for a model whose task is
  `next_token_generation`; the prompt is the data block of `eval.json`.

## What phase 7 decided and left open

- **Every expert runs on every token.** A graph is traced for fixed shapes, and the rows an
  expert would take, and their count, come from the data. `top_k` therefore masks the gate; it
  does not save the experts' work. Running an expert on its own rows needs an index of the rows
  a mask selects, which SushiBLAS does not have.
- **The experts are one stacked feed-forward**: one linear layer to `E * H`, the gate widened
  over it, one projection back. It is 31 nodes for any count of experts, where a child per
  expert would be 22 + 10 E.
- **`top_k` is exact, from operations the graph had**: a row maximum through a pooling window,
  a comparison and a select, repeated per level. A tie keeps more than k. No gradient passes
  through the mask.
- **The balance term is the squared coefficient of variation of the mean routing
  probability**: 0 at an equal split, `E - 1` at collapse. It reads the softmax of every score,
  also under `top_k`, so its gradient reaches an expert that no position keeps.
- **A block with experts publishes three ports**, `output`, `balance` and `usage`. A training
  file sums `next_token` over the `output` it names with `expert_balance`, which reports
  `expert_usage_min` and `expert_usage_max`.
- **Inside a sum, perplexity is the exponential of the whole loss**, the balance included.
- **Every number of the training test is unverified**: 200 updates, a last loss under half the
  first, a least usage above 0.10 with the balance at 0.5.

## The decision model

Jev's weights, architecture paper and training method (named RLCD) are not published, so this
is a model of the same kind and not a copy of it.

| Part | Module | Holds |
| --- | --- | --- |
| Encoder | `blocks` | The transformer block of phase 2 without the causal mask, and a pooled output |
| Typed heads | `blocks` | One head per answer type over the pooled output: a choice from a list, yes or no, a number |
| Objectives | `training` | Those of [`DATASETS.md`](DATASETS.md): classification, binary, regression, one per head, summed with stated weights |
| Calibration | `training` | Temperature scaling fitted on a held-out split, and the expected calibration error as a metric |
| Pretraining | `training`, `datasets` | The masked-token objective and its collator |

A choice whose options change from one question to the next cannot be a fixed output layer.
The head scores each option's encoding against the pooled state, so the answer space is an
input and the model cannot return a value outside it.

It needs no generation loop and no key and value cache, so it does not wait on phase 5.

Ruled by the owner on 2026-10-11: the weights are exported to SushiEngine and run there; the
model is small enough for inference on a phone; an academic counterpart is the reference. The
phase is planned and waits until vision and the phases before it are finished.

## Open

Each is the owner's decision unless it says otherwise.

- **The corpus of the first run.** It is read from disk, as every dataset is.
- **Positions**: a learned table, which is what the reference scale uses, or rotary positions.
  The engineer's choice is the learned table first, since it needs no new operation.
- **The tokenizer's file format** and whether a vocabulary trained elsewhere is imported
  through the door of [`MODEL_IMPORT.md`](MODEL_IMPORT.md).
- **The decision model's frame budget and its training data.** Not stated yet.
- **The multi-machine design.** It needs four facts: the device of each machine, the network
  between them, their operating systems, and whether the distributed layer has run on them.
