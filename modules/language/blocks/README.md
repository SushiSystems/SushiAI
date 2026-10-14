# blocks

## Owns

The model blocks of a transformer over a token sequence, each built from `nn` layers:
`PositionEmbedding`, a learned table added to each position; `CausalSelfAttention`, a
multi-head self-attention; `FeedForward`, two linear layers around a GELU; `TransformerBlock`,
an attention and a feed-forward, each normalised first and added to what it read; and
`LanguageHead`, the final normalisation and the linear layer to the vocabulary. Every block is an `NN::TraceModule` over a `[N, T, C]` value, and every block has a
factory that builds an `NN::Node` from a row's attributes. A per-sample shape is `[T, C]`.

`RotaryEmbedding` turns the channel pairs of a sequence by the angle of each position. It
holds no state, and `CausalSelfAttention` traces it over its queries and keys when its
`positions` are rotary, in place of a `PositionEmbedding` row before the blocks.

Two more blocks hold experts: `MoEFeedForward`, a router and `E` feed-forward experts whose
results are summed by the gate the router gives each position, and `ExpertTransformerBlock`,
a `TransformerBlock` with that mixture in place of its feed-forward. Each traces three values,
its result, a balance term and the usage of its experts, and its node has one port for each.

A block refuses a description or an input that does not fit by recording the failure on the
trace and returning its input. No constructor throws. `trace_experts` of the two blocks with
experts returns the failure and records nothing; their `forward` records it.

## Depends on

- `core`, `graph` and `nn` (`PUBLIC_DEPENDS`).
- `config` (`PRIVATE_DEPENDS`).

## Public entry points

Consumers include each header by the path in the first column.

| Header | Declares |
| --- | --- |
| `<SushiAI/language/blocks/position_embedding.hpp>` | `PositionEmbedding`, `make_position_embedding_node` |
| `<SushiAI/language/blocks/causal_self_attention.hpp>` | `AttentionMask`, `AttentionPositions`, `CausalSelfAttention`, `check_attention_widths`, `check_attention_positions`, `make_causal_self_attention_node` |
| `<SushiAI/language/blocks/rotary_embedding.hpp>` | `RotaryEmbedding`, `check_rotary_widths` |
| `<SushiAI/language/blocks/rotary_positions.hpp>` | `K_ROTARY_BASE`, `rotary_angle`, `write_rotary_row` |
| `<SushiAI/language/blocks/feed_forward.hpp>` | `FeedForward`, `make_feed_forward_node` |
| `<SushiAI/language/blocks/transformer_block.hpp>` | `TransformerBlock`, `make_transformer_block_node` |
| `<SushiAI/language/blocks/language_head.hpp>` | `LanguageHead`, `make_language_head_node` |
| `<SushiAI/language/blocks/moe_feed_forward.hpp>` | `MoEFeedForward`, `check_expert_counts`, `make_moe_feed_forward_node` |
| `<SushiAI/language/blocks/expert_projection.hpp>` | `ExpertProjection` |
| `<SushiAI/language/blocks/expert_transformer_block.hpp>` | `ExpertTransformerBlock` |
| `<SushiAI/language/blocks/expert_trace.hpp>` | `ExpertTrace`, `K_BALANCE_PORT`, `K_USAGE_PORT` |
| `<SushiAI/language/blocks/incremental.hpp>` | `K_POSITION_LEAF`, `K_ATTENTION_MASK_LEAF`, `K_MASKED_ATTENTION_SCORE`, `K_CACHE_SLOT_LEAF`, `K_ROTARY_COSINE_LEAF`, `K_ROTARY_SINE_LEAF` |

Each `make_*_node` function takes the input shapes, the attributes and a context string, and
returns `Result<std::unique_ptr<NN::Node>>`. Each reads one per-sample `[T, C]` shape and takes
`C` as the width. `make_language_head_node` takes a fourth argument, the `NN::BuiltRows` above
its row, so it is an `NN::LinkedNodeFactory` where the others are an `NN::NodeFactory`; the
argument defaults to a view of no row.

### Attribute keys

| Factory | Row type it is written for | Required | Optional |
| --- | --- | --- | --- |
| `make_position_embedding_node` | `position_embedding` | | `length`, `T` when absent; at least `T` |
| `make_causal_self_attention_node` | `causal_self_attention` | `heads` | `causal`, true when absent; `positions`, `"learned"` when absent, or `"rotary"`, which needs `C / heads` even |
| `make_feed_forward_node` | `feed_forward` | `hidden` | |
| `make_transformer_block_node` | `transformer_block` | `heads` | `hidden`, four times the width when absent; `causal`, true when absent; `ln_epsilon`, 1e-5 when absent; `positions`, which its attention takes; `experts`, 0 when absent, which above 0 makes `hidden` the width of one expert and the node that of an `ExpertTransformerBlock`; `top_k`, 0 when absent, stated only with `experts` |
| `make_moe_feed_forward_node` | `moe_feed_forward` | `experts`, `hidden` | `top_k`, 0 when absent: every expert is kept; at most `experts` |
| `make_language_head_node` | `language_head` | `vocabulary`, unless `tie` is stated | `ln_epsilon`, 1e-5 when absent; `tie`, the `embedding` row above whose table the head scores with |

`modules/language/registration` adds the six factories to a node registry. `RotaryEmbedding`
has no row: a model states `"positions": "rotary"` on the rows that hold an attention and
leaves the `position_embedding` row out.

## The incremental forms

Every block has `forward_incremental(trace, x, sample)` beside `forward`, so a `ModelGraph`
of these rows traces through `ModelGraph::trace_incremental`. `x` is the `[1, 1, dim]` slice
of one sequence at one position and `sample` is the per-sample `[T, dim]` shape of the
ordinary form, whose `T` is the row count of a cache. The forms need a carrying trace,
`NN::Trace::carrying`.

| Block | One position | Declares |
| --- | --- | --- |
| `PositionEmbedding` | `x` plus `gather_rows(table, position)` | The fed leaf `K_POSITION_LEAF`, `[1, 1]` int32 |
| `CausalSelfAttention` | See below | `K_POSITION_LEAF`; the fed leaf `K_ATTENTION_MASK_LEAF`, `[1, T]`; two carried `[T, dim]` values, the keys and the values |
| `CausalSelfAttention`, rotary | See "The ring of a rotary attention" | The fed leaf `K_CACHE_SLOT_LEAF`, `[1, 1]` int32, in place of `K_POSITION_LEAF`; the mask; what its `RotaryEmbedding` declares; the two carried values |
| `RotaryEmbedding` | The turn of `x` by the fed angles | The fed leaves `K_ROTARY_COSINE_LEAF` and `K_ROTARY_SINE_LEAF`, `[1, pairs]` |
| `FeedForward`, `LanguageHead` | `forward`, since a position reads no other | Nothing |
| `TransformerBlock` | The incremental form of each child, in the order of `forward` | What its attention declares |
| `MoEFeedForward` | `trace_experts`, since the router of a position reads no other | Nothing |
| `ExpertTransformerBlock` | The incremental form of each child, in the order of `trace_experts` | What its attention declares |

A fed leaf is declared once per trace, so every block of a model reads one position leaf and
one mask leaf. The carried values are leaves of the step and no `NN::Buffer`: they are no
entry of the model state, and a block holds nothing between two traces.

The attention computes `q`, `k` and `v` of the position from `qkv`. It writes each cache as
`cache + scatter_add_rows(row, position, T)`, a zero `[T, dim]` with the row at the position
added to the cache, and carries that sum. The sum is the cache with one row written only
while the row holds zeros, which a reset and positions fed in order give. Per head it then
takes `softmax(q . keys^T / sqrt(dim / heads) + mask) . values` over all `T` rows and joins
the heads through `proj`. The mask is 0 at the positions up to the fed one and
`K_MASKED_ATTENTION_SCORE` beyond it, the score `causal_mask` writes above the diagonal in the
ordinary form; a row beyond the position holds zeros, so its score is the mask alone.

For `h` heads one position adds `20 + 15 h` nodes and eight leaves, where the ordinary form
adds `11 + 15 h` and four. The count per head is the same; the extents are not: a head scores
`[1, T]` where the ordinary form scores `[T, T]`. A block whose mask is `AttentionMask::None`
has no incremental form, since its position reads the positions after it.

The row write costs one `[T, dim]` scatter and one add per cache in the graph, and the carry
costs a scale and an axpy over `[T, dim]` behind it: four passes over a cache per run. The
attention reads `[T, dim]` keys and values in the same run, so the write does not change the
order of the cost. An operation that writes one row of a leaf in place, before the heads read
it, would remove the four passes; the graph has none.

### The ring of a rotary attention

A learned table gives token `i` of the window the row `i`, so a window that slides gives
every kept token another position and no cached key stays right. A rotary key is turned once,
by the position of its token counted from the start of the text, and the score of a query at
`p` with a key at `s` depends on `p - s` alone (see `rotary_embedding.hpp` below). The key of
a token therefore stays right for as long as the token is kept, and a slide is the write of
one row over the oldest one.

The run of the token at position `p` is fed the row `p mod T` under `K_CACHE_SLOT_LEAF`, the
cosines and the sines of `p`, and a mask that is 0 at every row a fed token fills. The rows
of a full ring are in no order of position, and need none: a key carries its position in its
own turn, and the softmax and the product with the values are sums over the rows, whichever
row holds which token.

A row that holds an older token has to be replaced, where a learned cache adds into zeros.
The block marks the row with `scatter_add_rows` of a `[1, 1, dim]` of ones, a `[T, dim]` that
is 1 in the row and 0 elsewhere, shared by both caches, and takes
`select(mark, scatter_add_rows(row, slot, T), cache)`: the new row where the mark is set and
the cache as it was elsewhere. No element is computed from the row it replaces, so the write
is exact. It costs five passes over `[T, dim]` in the graph for the two caches, the mark and
a scatter and a select each, against four for the two adds of a learned cache, and the same
four passes of the carry behind it.

For `h` heads one rotary position adds `43 + 15 h` nodes and ten leaves: the 21 nodes of the
turn, the ones and the mark, and the two angle leaves, which every block of a trace shares.

What the ring holds for a model of more than one block is stated in the README of
`language/generation`: a key of the second block was computed from what its token could read
when it was fed, which a slide does not recompute.

## Notes on the blocks

### position_embedding.hpp

`PositionEmbedding` adds row `t` of a `[length, dim]` parameter to position `t` of a
`[N, T, dim]` value. The graph has no broadcasting add, so the block sees the input as
`[N, T * dim]` rows and the table as one `[T * dim]` vector and emits one `BIAS_ADD`, which
adds a vector to every row of a matrix. A trace over `T = length` adds 5 nodes: the leaf of
the table, a view of each operand, the sum and the view of the result. The other forms cost
more for the same values: `channel_affine` multiplies by a gain of ones, and a gather by a
position index gives the `[T, dim]` rows the table already is.

A sequence shorter than the table reads its first `T` rows: the table is seen as
`[1, 1, 1, length * dim]`, cut by `slice_channels` to its first `T * dim` channels and seen as
a vector, which adds 2 nodes. A sequence longer than the table is refused with both lengths,
and no node is added.

The table has the role `Embedding` and is drawn from a normal of `NN::K_EMBEDDING_DEVIATION`,
so weight decay does not reach it.

### causal_self_attention.hpp

`CausalSelfAttention` traces `qkv`, a `Linear` from `dim` to `3 * dim`, over the `[N * T, dim]`
rows of its input. The result holds every query in its first `dim` channels, every key in the
next `dim` and every value in the last, and each of the three is cut into `heads` parts of
`dim / heads` channels. Each head multiplies its queries with its keys transposed, scales the
`[N, T, T]` scores by `1 / sqrt(dim / heads)`, replaces the entries above the diagonal through
`GraphBuilder::causal_mask`, takes the softmax over the last axis and multiplies the result with
its values. The heads are joined and `proj`, a `Linear` from `dim` to `dim`, is traced over
them. `AttentionMask::None` leaves the mask out, which is what an encoder asks for.

A trace of `H` heads adds 15 nodes around its heads, four of them the leaves of the two linear
layers, and 15 per head, or 14 per head without the mask. The channel emitters read rank 4, so
the projected value stands as a `[N, T, 1, 3 * dim]` map while it is cut.

The block differentiates through `K_LOG_SOFTMAX_BACKWARD`, one node per head: its softmax is
`RESHAPE`, `LOG_SOFTMAX`, `EXP`, `RESHAPE`.

With `AttentionPositions::Rotary` the block cuts the queries and the keys, the first
`2 * dim` channels of the projection, turns them through one `RotaryEmbedding(2 * dim,
2 * heads)` and cuts the heads from the result. The values are not turned. Every head turns
by the same angles, so one turn serves all of them: a rotary trace adds 21 nodes and two
table leaves to the counts above, `32 + 15 H` nodes and six leaves for a masked block,
whatever `H` is. `check_attention_positions` refuses a head of an odd number of channels.

### rotary_embedding.hpp

`RotaryEmbedding(width, groups)` cuts the channels of a `[N, T, width]` value into `groups`
equal parts and turns each part in pairs. Pair `i` of a part is channel `i` of its first
half, `a`, with channel `i` of its second half, `b`. At position `p` it becomes
`(a cos - b sin, a sin + b cos)` of the angle `p * 10000^(-i / pairs)`, which `rotary_angle`
returns; `pairs` is half the channels of a part. Every part turns by the same angles. The
layout is the same in both forms and in the fed rows.

Two vectors turned this way keep their product when both positions move by the same amount:
with `q = a + i b` and `k = c + i d` per pair, the product of `q` at `p` and `k` at `s` is the
real part of `q conj(k) e^(i (p - s) theta)`, a function of `p - s`.

The IR has no sine and no cosine, and the block needs none, since an angle depends on the
position alone: the host computes the cosines and the sines. The block reads them as two
`[rows, pairs]` values. With `swapped(x)` the value whose two halves of every part are
exchanged, the result is `x * (cos, cos) + swapped(x) * (-sin, sin)`. The halves of all parts
are exchanged at once, by viewing the value as `[N, T, groups, width / groups]` and joining
its two channel slices in the other order, so there is no permutation matrix and no node per
part. Each weight is a number per position and channel, and no elementwise operation
broadcasts: the value is viewed as `[N, 1, 1, T * width]` and each product is one
`channel_affine` with a shift of zeros, the multiplicative sibling of the `BIAS_ADD` of
`PositionEmbedding`. A trace adds 18 nodes and two leaves, whatever the group count:

| Nodes | What |
| --- | --- |
| 8 | The weights: a view of each angle value, the negated sines, a join and a view per weight, the zeros |
| 4 | The swap: a view, two channel slices, a join |
| 6 | The turn: a view of each operand, two `channel_affine`, the sum, the view of the result |

| Form | The angles are | Declared through |
| --- | --- | --- |
| `forward` | Two `[T, pairs]` tables for positions 0 to `T - 1` | `NN::Trace::host_table` |
| `forward_incremental` | Two `[1, pairs]` rows of the position of the fed token | `NN::Trace::feed`, under `K_ROTARY_COSINE_LEAF` and `K_ROTARY_SINE_LEAF` |

A host table is a leaf of the step, written once when the step is built. It is no
`NN::Parameter` and no `NN::Buffer`, so it is no entry of the model state, no optimizer slot
and no entry of a checkpoint. A table is float32, or float64 for a float64 value; a half
value reads a float32 table through a cast. A trace whose builder named no sink for tables,
which `Train`'s graph builders all do, refuses the block.

`write_rotary_row` fills the two fed rows for a position, and is the function a caller hands
to `Generation::CachedGenerator`. Both it and the tables call `rotary_angle`, so the two
forms read the same float32 numbers. The fed leaves are named, so the blocks of one trace
share them; blocks whose heads differ in width declare them with two shapes and the trace
refuses the second.

An angle is `p * theta` in double precision, whose rounding error is below `p * 2^-52`
radians. It stays under `2^-24`, the resolution of a float32 cosine near 1, while `p` is
below `2^28`, about 268 million tokens. Past that the angles lose precision gradually and
nothing refuses the position.

### feed_forward.hpp

`FeedForward` traces `expand`, a `Linear` from `dim` to `hidden`, a `GELU` and `project`, a
`Linear` back to `dim`, over the `[N * T, dim]` rows of its input. The GELU is the tanh
approximation `nn` has. A trace adds 11 nodes, four of them leaves.

### transformer_block.hpp

`TransformerBlock` traces `x + attn(norm1(x))`, then that sum plus `ff(norm2(sum))`. Its state is
listed under "norm1", "attn", "norm2" and "ff". The mask and the positions of the attention
are constructor arguments, so the same block serves a decoder and an encoder, over a learned
table or with rotary positions. A child that refuses stops the
trace, and the block returns its input.

### moe_feed_forward.hpp

`MoEFeedForward` holds `router`, a `Linear` from `dim` to `E` scores, and `E` experts, each a
feed-forward of hidden width `H`. For a row `x` of the `[N * T, dim]` rows of its input it
computes

```
y = sum over e of gate[e] * (GELU(x . W1_e + b1_e) . W2_e + b2_e)
```

Every expert runs on every row. A gate multiplies the hidden values of its expert before the
second layer, which gives the same sum, so the experts are two matrices and no list of
children. `expand` is a `Linear` from `dim` to `E * H` whose outputs `e * H` to
`e * H + H - 1` belong to expert `e`. `project` is an `ExpertProjection`: it holds the
`[dim, E * H]` weight and the `[E, dim]` bias of the second layers and emits
`gated . weight^T + gate . bias`. It declares its weight with the fans of one expert, so each
expert starts as a `FeedForward` of width `H` does. The `[N * T, E]` gate reaches the
`[N * T, E * H]` hidden values as a `[N * T, 1, E, 1]` map that `upsample2d` repeats `H` times
along its width.

A trace adds 31 nodes, six of them leaves, whatever `E` is. `E` separate `FeedForward`
children, joined by `concat_channels` and mixed by a rank-3 `matmul`, would add `22 + 10 E`:
62 for four experts and 102 for eight. A form over an `[E, dim, H]` parameter through the
rank-3 `matmul` needs the rows repeated `E` times along a leading axis, and the graph has no
emitter that does that.

**The gate.** With `top_k` of 0 the gate is the softmax of the scores. With `top_k` of `k` it
is the softmax over the `k` largest scores of the row, and every other expert has a gate of 0.
The graph has no top-k, no argmax and no sort, so `keep_top_scores` in
`source/top_scores.hpp` builds the mask from what it has:

| Step | Emitter | Result |
| --- | --- | --- |
| 1 | `reshape` | The scores as a `[N * T, 1, E, 1]` map, one image row per position |
| 2 | `maxpool2d`, window 1 by `E` | `[N * T, 1, 1, 1]`, the largest score of each row |
| 3 | `upsample2d`, 1 by `E` | That maximum under every score of its row |
| 4 | `less_equal` | 1 where the score is at or above it |
| 5 | `select` | The scores that are kept, and -1e9 in place of the others |

A further level repeats steps 2 to 4 over the scores with the kept ones replaced by -1e9, so
level `k` keeps every score at or above the `k`-th largest distinct value of its row. Equal
scores are kept together: a tie at the threshold keeps more than `k` experts. `less_equal` is
a gradient barrier, so the mask passes no gradient, and a kept score passes its own through
`select`. `top_k` of `k` adds `7 + 4 k` nodes: 42 in all for `k` of 1 and 46 for `k` of 2. The
experts a position drops are still computed, and their gate is 0. -1e9 is minus infinity in
half precision, as the masked score of the attention is.

**The balance term.** `usage` is the mean over the rows of the softmax of every score, a
`[1, E]` value, and `balance` is its squared coefficient of variation,
`E * sum(usage^2) - 1`, a `[1, 1]` value: 0 when every expert has the share `1 / E`, and
`E - 1` when one expert has all of it. Both read the softmax of every score, also when `top_k`
drops some, so the gradient of the balance reaches an expert that no position keeps.

`trace_experts` returns the three values as an `ExpertTrace`, and `forward` returns the result
alone. The node of a `moe_feed_forward` row has the ports `output`, `balance` and `usage`. A
port is a per-sample shape behind a batch axis, and the last two have no batch, so their
values stand behind an axis of 1. A model file publishes them by port, each under a name that
begins with the port's name; `expert_balance` of `language/training` finds them by that:

```json
"outputs": [
    { "node": 4 },
    { "node": 2, "port": "balance", "name": "balance2" },
    { "node": 2, "port": "usage", "name": "usage2" }
]
```

A port that no entry lists is not published, and the state of a block does not depend on what
is published, so a file without the two entries loads the same weights.

**What the block does not do.** It does not run an expert on its own positions alone. That
takes the rows of each expert gathered by an index whose values and whose count come from the
data, and a graph is traced once for fixed shapes. With a capacity `C` per expert fixed at
trace time the shapes would be fixed again: an operation that writes, for one expert, the row
numbers of the first `C` positions that keep it and a number outside the table in the unused
slots; `gather_rows` by that index; the expert over `[C, dim]`; and `scatter_add_rows` back
into `[N * T, dim]`. The two row operations exist. The index operation does not, in the graph
or in SushiBLAS, and the gather has no defined result for an index outside its table.

### expert_transformer_block.hpp

`ExpertTransformerBlock` traces `x + attn(norm1(x))`, then that sum plus the mixture over
`norm2` of it, and hands on the usage and the balance of its `ff`. Its state is listed under
"norm1", "attn", "norm2" and "ff", and under "ff" as "router", "expand" and "project". The two
branches share `add_normalised_branch` of `source/normalised_branch.hpp` with
`TransformerBlock`. A causal trace of two heads adds 84 nodes where a `TransformerBlock` adds
64. The block is built from a `MoEFeedForward`, whose width is that of the attention and of
the two normalisations. Its last constructor argument is the positions of its attention, as
it is for `TransformerBlock`, and `make_transformer_block_node` hands `positions` to
whichever of the two it builds.

A `transformer_block` row that states `experts` above 0 builds this block behind the node of
three ports, `Detail::ExpertNode` in `source/expert_node.hpp`; a row that states none builds a
`TransformerBlock` behind an `NN::ModuleNode` of one port. A row that reads such a row reads
its `output` port.

### language_head.hpp

`LanguageHead` traces `norm`, a `LayerNorm`, then `linear`, a `Linear` from `dim` to `vocabulary`
with no bias, over the rows, and returns `[N, T, vocabulary]` logits. A trace adds 7 nodes, three
of them leaves.

`tie_to(table)` makes the head score with a `[vocabulary, dim]` parameter another layer owns,
which is the layout of an `NN::Embedding` table and of a `Linear` weight alike. The tied head
declares the table's own `Parameter` again, which returns the leaf the graph already holds,
and multiplies the rows by it transposed. `linear.weight` is then never declared, so the walk
yields the table once, under the embedding's name, and a model has one state entry, one
gradient and one optimizer slot for it. A trace adds 6 nodes. The head refuses a table no
layer has declared yet and one of another shape or element type.

The head keeps the address of the table. A model that is copied or moved after `tie_to` is
tied to the object it was copied from, so a model ties in its constructor and forbids the
copy, as the decoder of `tests/language/training/test_next_token_objective.cpp` does.

A `language_head` row of a model file ties with `"tie"`, a reference to the `embedding` row
above it in the forms `from` takes for a row: a row number, `-1` for the row before, or
`{ "node": 0 }`. The factory finds the `NN::Embedding` of that row through
`NN::find_embedding` and calls `tie_to` on its weight. The table lives in its node, which the
`ModelGraph` owns on the heap, so the address holds when the graph moves. A row that is not
above, a row that is no `embedding` row, and a table that is not `[vocabulary, dim]` are each
refused when the model is built, with the row's context. A tied row may leave out
`vocabulary`, which is then the table's row count.

Source: `docs/design/LANGUAGE.md`, phase 2, and phase 7 for the two blocks with experts.
