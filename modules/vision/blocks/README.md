# blocks

## Owns

The model blocks of the YOLO family, each built from `nn` layers: `Conv`, a convolution with a
batch normalisation and a SiLU; `ConvLinear`, the same without the activation; `Bottleneck`;
`C3k`; `C3`, a `C3k` with a stated hidden width and a 1x1 then a 3x3 window in each
bottleneck; `C3k2`, which takes a `Bottleneck`, a `C3k` or a `Bottleneck` with a `PSABlock` as
its inner block; `C2f`, a `C3k2` over bottlenecks of the full hidden width; `SPPF`; `DWConv` and `DWConvLinear`, the depthwise forms of `Conv` and `ConvLinear`;
`Attention`, a multi-head self-attention over the positions of a feature map; `PSABlock`, an
attention with a feed-forward pair; `C2PSA`, which chains `PSABlock`s; `DetectHead`, the
detection head; and `SegmentHead`, the semantic segmentation head. Every block but `DetectHead`
is a `NN::TraceModule`: it traces itself into an
`NN::Trace` and computes the shape it produces. `DetectHead` is an `NN::Node` with one input per
feature level and four ports per level. Every block a model file can name has a factory that
builds an `NN::Node` from a row's attributes, and the comment of each factory lists the keys it
reads. `Attention` and `PSABlock` have none.

## Depends on

- `config`, `failure`, `shape`, `graph` and `nn` (`PUBLIC_DEPENDS`).

## Public entry points

Consumers include each header by the path in the first column.

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/blocks/conv_norm.hpp>` | `ConvNorm`, `Conv`, `ConvLinear`, `DWConv`, `DWConvLinear`, `make_conv_node` |
| `<SushiAI/vision/blocks/bottleneck.hpp>` | `Bottleneck`, `make_bottleneck_node` |
| `<SushiAI/vision/blocks/c3k.hpp>` | `C3k`, `make_c3k_node` |
| `<SushiAI/vision/blocks/c3.hpp>` | `C3`, `make_c3_node` |
| `<SushiAI/vision/blocks/c3k2.hpp>` | `C3k2`, `C3k2InnerEntry`, `K_C3K2_INNER`, `make_c3k2_of_bottleneck`, `make_c3k2_of_c3k`, `make_c3k2_of_bottleneck_psa`, `make_c3k2_node` |
| `<SushiAI/vision/blocks/c2f.hpp>` | `C2f`, `make_c2f_node` |
| `<SushiAI/vision/blocks/sppf.hpp>` | `SPPF`, `make_sppf_node` |
| `<SushiAI/vision/blocks/attention.hpp>` | `Attention` |
| `<SushiAI/vision/blocks/psa_block.hpp>` | `PSABlock` |
| `<SushiAI/vision/blocks/c2psa.hpp>` | `C2PSA`, `make_c2psa_node` |
| `<SushiAI/vision/blocks/detect_head.hpp>` | `BoxBranch`, `ClassStage`, `ClassBranch`, `DetectHead`, `make_detect_node` |
| `<SushiAI/vision/blocks/segment_head.hpp>` | `SegmentHead`, `make_segment_head_node` |

Each `make_*_node` function takes the input shapes, the attributes and a context string, and
returns `Result<std::unique_ptr<NN::Node>>`. `make_c3k2_node` reads the `inner` key and selects
one row of `K_C3K2_INNER`; it fails when the key names no row.

### Attribute keys

The keys each factory reads, as its comment states them. The default of each optional key is in
that comment.

| Factory | Required | Optional |
| --- | --- | --- |
| `make_conv_node` | `filters` | `kernel`, `stride`, `bn_epsilon`, `bn_momentum` |
| `make_bottleneck_node` | `filters`, `hidden` | `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_c3k_node` | `filters` | `repeats`, `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_c3k2_node` | `filters`, `hidden` | `inner`, `repeats`, `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_c3_node` | `filters`, `hidden` | `repeats`, `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_c2f_node` | `filters`, `hidden` | `repeats`, `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_sppf_node` | `filters` | `kernel`, `pools`, `shortcut`, `bn_epsilon`, `bn_momentum` |
| `make_c2psa_node` | `filters`, `hidden` | `repeats`, `heads`, `bn_epsilon`, `bn_momentum` |
| `make_detect_node` | `classes` | `bn_epsilon`, `bn_momentum` |
| `make_segment_head_node` | `classes` | `hidden`, `scale`, `bn_epsilon`, `bn_momentum` |

`registration` adds the factories of `conv`, `bottleneck`, `c3k`, `c3k2`, `c2f`, `c3`, `sppf`,
`c2psa`, `detect` and `segment_head` to the node registry.

## Notes on the blocks

The first five headers are the composite blocks of YOLO26; `c3.hpp` and `c2f.hpp` add two rows
over the same blocks. Each of their blocks is an
`NN::TraceModule`, and each has a factory with the `NodeFactory` signature. Source for those five:
MG 2.10, with the state names of MG 2.11. PLAN unit U11. The other four hold the attention blocks
and the head; their source is `docs/agent/2026_10_07_YOLO26/PLAN.md`, units U6 to U9.

### conv_norm.hpp

`ConvNorm<Convolution, Activation>` applies a bias-free convolution, a batch normalisation and
an activation, and lists its state under "conv" and "bn". `Conv` is `ConvNorm<Conv2d, SiLU>`
and `ConvLinear` is `ConvNorm<Conv2d, Identity>`. `DWConv` and `DWConvLinear` are the same two
over `DepthwiseConv2d`, which refuses unequal input and output widths. `make_conv_node` reads `filters`, the
optional `kernel` and `stride`, which are 1 when absent, and the two norm keys.

### bottleneck.hpp

`Bottleneck` traces `cv1`, then `cv2`, then an add of its input. The add is made only when a
shortcut is asked for and the input width equals the output width; `adds_input` tells which.
Both windows are 3 wide unless the caller states them. `make_bottleneck_node` reads `filters`
and `hidden`, and the optional `shortcut`, which is true when absent.

### c3k.hpp

`C3k` traces `cv1` and a chain of bottlenecks, then `cv2` on a second branch, then `cv3` over
the two results joined. The hidden width is half the output width, rounded down. The chain is
the `ModuleList<Bottleneck>` named `m`. `make_c3k_node` reads `filters` and the optional
`repeats`, 1 when absent, and `shortcut`, true when absent.

### c3k2.hpp

`C3k2<Inner>` traces `cv1`, cuts its result into two halves of `hidden` channels, chains the
inner blocks of `m` off the second half, and traces `cv2` over both halves and every inner
result. `K_C3K2_INNER` lists the inner blocks a `c3k2` row may name: "bottleneck", "c3k" and
"bottleneck_psa", a `Bottleneck` followed by a `PSABlock` of `max(hidden / 64, 1)` heads.
`make_c3k2_node` reads `filters` and `hidden`, and the optional `inner`, "bottleneck" when
absent, `repeats`, 1 when absent, and `shortcut`, true when absent.

### c3.hpp

`C3` is `C3k`. It is built through the constructor that states the hidden width and the two
windows of each bottleneck: `cv1` and `cv2` are 1x1 convolutions from the input width to `hidden`,
each bottleneck keeps `hidden` channels with a 1x1 window then a 3x3 window, and `cv3` maps the
chain and `cv2` joined, `2 * hidden` channels, to the output width. The state names are those of
`C3k`: `cv1`, `cv2`, `cv3` and `m.<i>.cv1`, `m.<i>.cv2`. `make_c3_node` reads `filters` and
`hidden`, and the optional `repeats`, 1 when absent, and `shortcut`, true when absent.

### c2f.hpp

`C2f` is `C3k2<Bottleneck>`. `cv1` is a 1x1 convolution to `2 * hidden` channels, cut into two
halves; `repeats` bottlenecks of `hidden` channels with 3x3 windows are chained off the second
half; `cv2` maps the two halves and every bottleneck result, `(2 + repeats) * hidden` channels, to
the output width. The state names are `cv1`, `cv2` and `m.<i>.cv1`, `m.<i>.cv2`. The bottleneck
of the "bottleneck" inner block of `c3k2` has a hidden width of `hidden / 2`; here it is `hidden`.
`make_c2f_node` reads `filters` and `hidden`, and the optional `repeats`, 1 when absent, and
`shortcut`, true when absent.

### sppf.hpp

`SPPF` traces `cv1`, a `ConvLinear` to half the input width, then a stride-1 max pool applied
`pools` times in a chain, then `cv2` over `cv1`'s result and every pooled value joined, then
an add of its input when `adds_input` holds. `make_sppf_node` reads `filters` and the optional
`kernel`, 5 when absent, `pools`, 3 when absent, and `shortcut`, false when absent. The kernel
must be odd.

### attention.hpp

`Attention` traces `qkv`, a `ConvLinear` to `channels + 2 * heads * key_dim` channels, and cuts
its result head by head into queries, keys and values. `head_dim` is `channels / heads` and
`key_dim` is `head_dim / 2`. Each head scales its queries by `1 / sqrt(key_dim)`, multiplies them
with its keys transposed, takes the softmax over the last axis and multiplies the result with its
values. The heads are joined, `pe`, a 3x3 `DWConvLinear` over the values of all heads, is added,
and `proj`, a `ConvLinear`, is traced over the sum. The constructor throws on a width the head
count does not divide, or on fewer than two channels per head. It has no factory of its own.

### psa_block.hpp

`PSABlock` traces `attn`, an `Attention` over its input, and adds the input to the result. It then
traces `ffn`, a `Conv` to twice the width followed by a `ConvLinear` back to the width, and adds
the sum to that result. Its state is listed under "attn" and "ffn". It has no factory of its own.

### c2psa.hpp

`C2PSA` traces `cv1`, a `Conv` to twice `hidden` channels, and keeps the first half. It chains
`repeats` `PSABlock`s off the second half, then traces `cv2`, a `Conv` from twice `hidden` to the
output width, over the kept half and the chained result joined. `make_c2psa_node` reads `filters`
and `hidden`, and the optional `repeats`, 1 when absent, `heads`, `max(hidden / 64, 1)` when
absent, and the two norm keys. It refuses a `filters` that is not the input width, and a `hidden`
that `heads` does not split into parts of at least two channels.

### detect_head.hpp

`DetectHead` holds two copies of the head, one-to-many under "cv2" and "cv3" and one-to-one under
"one2one_cv2" and "one2one_cv3". Per level, a `BoxBranch` is two 3x3 `Conv` and a biased 1x1
convolution to 4 channels, and a `ClassBranch` is two `ClassStage`s, each a 3x3 `DWConv` and a
1x1 `Conv`, and a biased 1x1 convolution to `classes` channels. The box branches are
`max(16, C0 / 4, 4)` wide and the class branches `max(C0, min(classes, 100))`, where `C0` is the
width of the first level. The ports are named `<copy>.level<l>.boxes` and `<copy>.level<l>.scores`:
the copy `one2many` first, and in each copy the boxes of every level, then the scores. A training
trace passes each feature through `stop_gradient` before the one-to-one copy; an evaluation trace
does not. `make_detect_node` reads `classes` and the two norm keys.

### segment_head.hpp

`SegmentHead` traces `cv1`, a 3x3 `Conv` to `hidden` channels, then `cv2`, a biased 1x1
convolution to `classes` channels, then `up`, an `NN::Upsample2d` by `scale`. Its output is
`[N, H * scale, W * scale, classes]` logits in NHWC, the form `pixel_cross_entropy` and `dice`
of `vision/training` read. Its state is listed under "cv1" and "cv2"; the upsample holds none.
`make_segment_head_node` reads `classes` and the optional `hidden`, the input width when absent,
`scale`, 1 when absent, and the two norm keys. It is the `segment_head` row.

`scale` is the product of the strides of the body, so that the logits end at the input size;
the factory of each objective refuses a model whose output has another height or width than its
input. The
head has no source in another implementation.

`NN::Upsample2d` repeats each pixel, and the graph has no bilinear upsample. Every
`scale` by `scale` block of the output therefore holds one logit vector, and a class boundary
can sit only on a multiple of `scale` pixels. With `scale` 8 a structure narrower than 8 pixels
is lost or widened to 8, which lowers IoU on thin classes; a head fed from a stride-2 or
stride-4 feature map loses less. A bilinear upsample needs an operation in `core/graph`.
