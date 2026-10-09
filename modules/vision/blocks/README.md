# blocks

## Owns

The model blocks of the YOLO family, each built from `nn` layers: `Conv`, a convolution with a
batch normalisation and a SiLU; `ConvLinear`, the same without the activation; `Bottleneck`;
`C3k`; `C3k2`, which takes a `Bottleneck` or a `C3k` as its inner block; and `SPPF`. Every block
is a `NN::TraceModule`: it traces itself into an `NN::Trace` and computes the shape it produces.
Every block also has a factory that builds an `NN::Node` from a model description's attributes,
and the comment of each factory lists the keys it reads.

## Depends on

- `core`, `graph` and `nn` (`PUBLIC_DEPENDS`).
- `config` (`PRIVATE_DEPENDS`).

## Public entry points

Consumers include each header by the path in the first column.

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/blocks/conv_norm.hpp>` | `ConvNorm`, `Conv`, `ConvLinear`, `make_conv_node` |
| `<SushiAI/vision/blocks/bottleneck.hpp>` | `Bottleneck`, `make_bottleneck_node` |
| `<SushiAI/vision/blocks/c3k.hpp>` | `C3k`, `make_c3k_node` |
| `<SushiAI/vision/blocks/c3k2.hpp>` | `C3k2`, `C3k2InnerEntry`, `K_C3K2_INNER`, `make_c3k2_of_bottleneck`, `make_c3k2_of_c3k`, `make_c3k2_node` |
| `<SushiAI/vision/blocks/sppf.hpp>` | `SPPF`, `make_sppf_node` |

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
| `make_sppf_node` | `filters` | `kernel`, `pools`, `shortcut`, `bn_epsilon`, `bn_momentum` |

`registration` adds the factories of `conv`, `bottleneck`, `c3k`, `c3k2` and `sppf` to the node
registry.

## Notes on the blocks

The five headers are the composite blocks of YOLO26. Each block is an `NN::TraceModule`, and each has a factory with the `NodeFactory` signature. Source for all five:
MG 2.10, with the state names of MG 2.11. PLAN unit U11.

### conv_norm.hpp

`ConvNorm<Convolution, Activation>` applies a bias-free convolution, a batch normalisation and
an activation, and lists its state under "conv" and "bn". `Conv` is `ConvNorm<Conv2d, SiLU>`
and `ConvLinear` is `ConvNorm<Conv2d, Identity>`. `make_conv_node` reads `filters`, the
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
result. `K_C3K2_INNER` lists the inner blocks a `c3k2` row may name: "bottleneck" and "c3k".
`make_c3k2_node` reads `filters` and `hidden`, and the optional `inner`, "bottleneck" when
absent, `repeats`, 1 when absent, and `shortcut`, true when absent.

### sppf.hpp

`SPPF` traces `cv1`, a `ConvLinear` to half the input width, then a stride-1 max pool applied
`pools` times in a chain, then `cv2` over `cv1`'s result and every pooled value joined, then
an add of its input when `adds_input` holds. `make_sppf_node` reads `filters` and the optional
`kernel`, 5 when absent, `pools`, 3 when absent, and `shortcut`, false when absent. The kernel
must be odd.
