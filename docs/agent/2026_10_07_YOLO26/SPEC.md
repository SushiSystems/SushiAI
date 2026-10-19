# YOLO26 on SushiAI: requirements map

**Status:** Open — the forward pass is built; RW-34 to RW-42.

This map lists what training Ultralytics YOLO26 from scratch, and running its forward pass, needs from SushiAI, SushiBLAS and SushiRuntime. Nothing was built or run for it. Every "exists" is a reading of source that has never executed (SPEC:3, SPEC:59-60). Ultralytics line numbers are from commit `ce1399439b93c490b648b55dedd0a6334f0199da`.

Abbreviations:

- **SushiAI** (`D:\Projects\sushiai`): `ops` = `include/SushiAI/core/ops.hpp`, `bh` / `bc` = `include/SushiAI/graph/builder.hpp` / `src/graph/builder.cpp`, `low` = `src/graph/lowering.cpp`, `diff` = `src/autograd/differentiator.cpp`, `SPEC` = `docs/agent/2026_10_07_CV_CLOSE/SPEC.md`. Other paths are relative to the repository root.
- **SushiBLAS** (`D:\Projects\sushiblas\include\SushiBLAS\engine`): `el` = `math/elementwise.hpp`, `nl` = `math/nonlinear.hpp`, `red` = `math/reductions.hpp`, `lg` = `logic/logic.hpp`, `sp` = `spatial.hpp`, `l1` = `blas/level1.hpp`, `l3` = `blas/level3.hpp`.
- **Ultralytics**: `U/` = `ultralytics/` at the commit above.

"Derived here" marks a composition or formula worked out on paper from the cited lines and not executed.

## 1. YOLO26 as primitive operations

### Forward pass

| Operation | Where YOLO26 uses it | SushiBLAS kernel | IR op / emitter | Gradient rule | Lowering |
| --- | --- | --- | --- | --- | --- |
| Conv, groups=1, no bias: 3x3 s2 p1, 3x3 s1 p1, 1x1 | `Conv` everywhere (U/nn/modules/conv.py) | `im2col` sp:59, `gemm` l3:51 | exists, `conv2d` bc:518-569 | exists (im2col, matmul, reshape) | exists low:1014, 446, 930 |
| Conv bias | last 1x1 of each head branch (U/nn/modules/head.py:125, 134) | `copy` + `ger` | exists, `BIAS_ADD` bc:135 | exists | exists low:460 |
| Depthwise conv 3x3, groups = C | `DWConv` in the class branch (head.py:132-133; `g = gcd(c1, c2)`, conv.py:209); `Attention.pe` (U/nn/modules/block.py:1309) | no grouped kernel: `ConvGeometry` has no group field (support/spatial_geometry.hpp:40-58) | no single op: `conv2d` forces weight C_in == input C (bc:533-537). Composable from `IM2COL`, `RESHAPE`, `CHANNEL_AFFINE`, `SLICE_CHANNELS`, `ADD` (below) | exists for every op of the composition | exists for every op of the composition (low:1014, 930, 1225, 1193, 689) |
| BatchNorm, batch statistics | every `Conv` | composed in the lowering | exists, `BATCHNORM2D` bc:873-910, eps in slot 0, outputs `y`, `mean`, `inv_std` (bc:909) | exists (src/autograd/rules/norm.cpp:34) | exists low:1239 |
| BN running mean and variance update, momentum 0.03 (U/utils/torch_utils.py:681-682) | every `Conv` in training | `axpy` l1:53, `scal` l1:65 | `BATCHNORM_UPDATE_STATS` is declared only (ops:191); no emitter. The batch variance is not a node output (below) | not needed | **missing** (falls into the throw at low:440) |
| BN eval | validation, inference | `channel_affine` sp:159 | `CHANNEL_AFFINE` exists bc:836; no fold of running statistics into gamma and beta | exists | exists low:1225 |
| SiLU | every `Conv` with act | exists: `silu` nl:143, `silu_backward(dy, x, dx)` nl:153 | **missing**: no op, no row in `ACTIVATIONS` (ops:322-328 has five rows) | missing | missing |
| Add, same shape | Bottleneck, SPPF, PSABlock, Attention | `add` el:60 | exists bc:180 | exists | exists low:689 |
| Concat on channels (2, 3 or 4 parts) | Concat rows, C2f, C3, SPPF, C2PSA | `concat_channels` sp:140 | exists bc:772 | exists | exists low:1172 |
| Split / chunk on channels | C2f, C2PSA, Attention q/k/v | `slice_channels` sp:149 | exists bc:814 | exists | exists low:1193 |
| Max pool 5x5 s1 p2 | SPPF, three in series (block.py:229-240) | `pool_max` sp:86; the running maximum starts at the lowest finite value, so padding is never a candidate (SushiBLAS engine/README.md:278-281) | exists bc:571 | exists | exists low:1050 |
| Nearest upsample x2 | rows 11, 14 | `upsample_nearest` sp:122 | exists bc:638 | exists | exists low:1110 |
| Reshape `[B,H,W,C]` <-> `[B,N,C]` | Attention, head | `copy` | exists bc:955 | exists | exists, as a copy (low:951); the source must be contiguous (low:938) |
| Batched matmul, rank 3, transpose flag | Attention, two per head (block.py:1331-1332) | exists: batched `gemm`, contiguous operands only, the output sets the batch count (l3:51; blas/README.md:270-282) | **missing**: `MATMUL` throws unless rank 2 (bc:105-107) | rank-2 rule only (src/autograd/rules/linalg.cpp:34) | calls `gemm` (low:455); batched shapes unhandled |
| Softmax, last axis | Attention (block.py:1331) | no single kernel | forward exists as `exp(log_softmax(reshape(x, [B*N, N])))`: `LOG_SOFTMAX` is rank 2 (low:819) and `EXP` exists. Not differentiable: `LOG_SOFTMAX` is `op()` (ops:199) | missing; the standalone rule throws (src/autograd/rules/loss.cpp:34-44) | forward exists (low:815-855, 757); backward missing |
| Scale by a constant | `q * self.scale` (block.py:1304, 1331) | `copy` + `scal` | exists, `SCALE` bc:195 | exists | exists low:707 |
| Stop-gradient | features into the one-to-one head (head.py:196) | none needed | **missing** (below) | missing: a registered rule that contributes nothing, or a barrier in `propagate_requires_grad` | missing (alias or copy) |
| Sigmoid | class scores at inference | exists nl:92 | exists | exists | exists low:760 |
| Box decode: anchor ± distance, times stride | `dist2bbox` (U/utils/tal.py:457-476), inference | add, sub, mul | exists if anchors and strides are full-shape leaves; binaries need equal shapes (bc:66-73) | not needed | exists |
| Top-k, gather, max over classes, index mod and div | `postprocess`, `get_topk_index` (head.py:249-288) | missing: no top-k, no gather, no axis argmax (`argmax` red:97 is whole-tensor) | missing | not needed | missing; runs as host code |

**Depthwise composition (derived here).** `im2col` orders a patch `(r, s, c)` with the channel innermost (SushiBLAS engine/README.md:246-248; bc:557), so column `t*C + c` is tap `t` of channel `c`. A 3x3 depthwise convolution is then:

1. `im2col_with_extents` to `[N*Ho*Wo, 9C]` (bc:446-465);
2. `reshape` to `[N, Ho, Wo, 9C]`;
3. `channel_affine` with the `[9C]` weight as gamma and a zero beta (bc:836-857); its rule yields the weight gradient as `sum_channels(mul(dy, x))` (norm.cpp:71-73);
4. nine `slice_channels` of width C and eight `add`.

The cost per depthwise layer is a `[N, Ho, Wo, 9C]` column tensor and an affine result of the same size. A depthwise kernel in SushiBLAS removes that cost and is a request under decision D4, not a precondition.

**Stop-gradient.** An operation without a rule on a gradient path throws: `requires_grad` propagates through every node unconditionally (include/SushiAI/graph/ir.hpp:423-434), and `differentiate_node` throws "No backward rule" for a reachable node without one (diff:199-204). A static assertion also requires a rule for every differentiable op (include/SushiAI/autograd/rules/all.hpp:72-80). Stop-gradient therefore needs one of two mechanisms, both host-lane:

- a `STOP_GRADIENT` op whose registered rule returns without contributing;
- a barrier at which `propagate_requires_grad` stops, so consumers never ask for the gradient.

### Differentiated loss

Reference: U/utils/loss.py:120-157, 420-498, 1322-1354, and `bbox_iou` at U/utils/metrics.py:108-170.

| Operation | Where | SushiBLAS kernel | IR op / emitter | Gradient rule | Lowering |
| --- | --- | --- | --- | --- | --- |
| Add, sub, mul, same shape | decode, CIoU, weighting, masking | el:60, 70, 80 | exists | exists | exists low:689 |
| Divide | `inter / union`, `rho2 / c2`, `w / h`, `v / (1 - iou + v + eps)` (metrics.py:157-169) | `div` el:90 | **missing** | missing: `da = dc / b`, `db = -dc * a / b^2` | missing |
| Binary min, max | CIoU intersection and enclosing box (metrics.py:149-159) | el:276, el:286 | **missing** | missing: the gradient goes to the selected operand, which needs a comparison mask; SushiBLAS has `less_equal` and `where` (lg:100, 105) and the functor seams | missing |
| Clamp at 0 | `.clamp_(0)` on intersection sides (metrics.py:149-151) | relu | `RELU` exists | exists | exists |
| atan | CIoU `v` (metrics.py:166) | el:202, in place | **missing** | missing: `dx = dy / (1 + x^2)` | missing |
| abs | L1 on ltrb (loss.py:152-154) | el:146, in place | **missing** | missing: `dx = dy * sign(x)`; SushiBLAS has no `sign` kernel | missing |
| BCE with logits, soft targets, no reduction | loss.py:358, 451-454 | `softplus` nl:179, `softplus_backward` nl:189 | **missing** (no softplus, no BCE) | missing: for `softplus(x) - x*t`, `dx = dy * (sigmoid(x) - t)` | missing |
| Sum of all elements to one element | three loss terms per level and branch | `gemv` with the transpose flag | exists: `sum_rows(reshape(x, [n, 1]))` yields shape `[1]` (bc:157-167, 955-966); `sum_channels` already uses this composition (bc:859-871) | exists (linalg.cpp:78-90) | exists (low:879-892); one reshape copy per term (low:951) |
| Seed of the loss gradient | the loss root is an `ADD` of `MUL`s, not `NLL_LOSS` | none | **missing**: `differentiate()` throws unless `NLL_LOSS` produces the root (diff:142-154) | every other rule returns on an absent output gradient (src/autograd/rules/elementwise.cpp:36-37) | none, or one constant |
| Stop-gradient inside an expression | CIoU `alpha` under `no_grad` (metrics.py:167-169) | none | **missing**, same mechanism as above | as above | missing |
| Multiply a 1-element value by a per-epoch scalar | `o2m`, `o2o` (loss.py:1344) | mul | `MUL` with a 1-element leaf exists; `DYNAMIC_SCALE` is `op()`, not differentiable (ops:122) | exists via `MUL` | exists |
| Slice the 4 box columns | `chunk(4, -1)` (metrics.py:143-144) | `slice_channels` | exists if boxes stay `[B,H,W,4]` | exists | exists |
| Row gather by foreground index | `pred_bboxes[fg_mask]` (loss.py:132-134) | missing | missing | missing | missing; not needed in the masked fixed-shape form (section 4) |
| Concat along the anchor axis | `forward_head` (head.py:176) | missing (channel axis of rank 4 only, bc:777-779) | missing | missing | missing; not needed with per-level loss terms |
| Permute (0,2,1) | loss input layout (loss.py:420-423) | no general permute | missing | missing | not needed in NHWC |

BCE written as `softplus(x) - x*t` is the standard identity; torch's implementation was not opened.

**Loss root.** `Differentiator::run` requires the producer of the loss value to be `NLL_LOSS` (diff:148-154), and `resolve` returns `NO_VALUE` for the root (diff:88). A detection loss needs a general seed: `differentiate()` contributes an explicit 1-element gradient to the root (one, times `seed_factor` when given; include/SushiAI/autograd/backward.hpp:54-61), `resolve` returns it like any other contribution, and the `NLL_LOSS` rule drops its root-only check (loss.cpp:50-53). The change is in `src/autograd/differentiator.cpp` and `src/autograd/rules/loss.cpp`, both host-lane. The seed value is either a 1-element leaf the trainer feeds with one, which needs no new lowering, or a constant op, which needs one.

### Label assignment

Reference: U/utils/tal.py:63-386, under `no_grad` (tal.py:63). No gradient rule is needed for any row.

| Operation | SushiBLAS | IR |
| --- | --- | --- |
| Comparisons, `where`, logical and | exist: `equal` to `less_equal`, `where`, `logical_and` (lg:55-114), tensor operands only | missing |
| `pow` with exponents 0.5 and 6.0 (loss.py:377-378) | `pow(t, double)` el:154 | missing |
| Top-k on the last axis (k = 10, 7, 1; loss.py:1327-1328, tal.py:373-376) | missing | missing |
| Scatter, `scatter_add`, advanced-index writes | missing | missing |
| `nonzero`, index-triple gather | missing | missing |
| Max and argmax along an axis of a rank-3 tensor | `max_along` is rank 2 only (red:152; math/README.md:613-621); `argmax` is whole-tensor (red:97) | missing |
| CIoU forward | same kernels as the loss | same gaps |
| One-hot by scatter (tal.py:306-311) | missing | missing |
| Enlarge ground-truth sides under `stride_val` (tal.py:334-340) | `where`, comparison | missing |

Run on the host (decision D2), none of this needs a kernel or an IR op.

## 2. Layers and blocks

"After CV" means present in the working tree per SPEC:43, unbuilt.

| Ultralytics module | Decomposes into | State after CV work |
| --- | --- | --- |
| `Conv` (conv + BN + SiLU, or no act) | `Conv2d` (no bias) + `BatchNorm2d` + `SiLU` | `Conv2d` exists. `BatchNorm2d` layer missing (SPEC:33). `SiLU` layer missing. |
| `DWConv` | depthwise conv + BN + SiLU | missing: no group count in `ConvGeometry` (bh:46-56) or `NN::Conv2d`; the composition of section 1 has no layer |
| `nn.Conv2d` 1x1 with bias | `Conv2d` with bias | exists; constant bias initialisers missing (2.0 for the box branch, `log(5 / nc / (640 / stride)^2)` for the class branch, head.py:229-241) |
| `Bottleneck` | 2x `Conv` + add | needs a block with a skip: missing (SPEC:68-69) |
| `C2f` / `C3k2` | `Conv`, slice, n inner blocks, concat, `Conv`; with `attn=True`, each inner block is `Bottleneck` then `PSABlock` (block.py:1095-1100, yaml row 22) | missing: multi-branch block |
| `C3` / `C3k` | 3x `Conv`, n `Bottleneck`, concat | missing |
| `SPPF` | `Conv` (no act), 3x `MaxPool2d`, concat of 4, `Conv`, add | `MaxPool2d` exists; the block is missing |
| `Attention` | `Conv` x2 (no act), depthwise `Conv` (no act), slices, reshape, rank-3 matmul, softmax, concat, add (block.py:1301-1334) | missing |
| `PSABlock`, `C2PSA` | `Attention`, 2x `Conv`, adds, slice, concat | missing |
| `nn.Upsample` | `Upsample2d` | exists (registry row `upsample`, include/SushiAI/nn/layer_registry.hpp:195) |
| `Concat` | `CONCAT_CHANNELS` | IR exists; no layer (SPEC:68) |
| `Detect` | per level and per head: box branch (2x `Conv` 3x3 + 1x1 with bias) and class branch (2x [`DWConv` 3x3 + `Conv` 1x1] + 1x1 with bias) (head.py:124-136); a second copy of both made by `deepcopy` (head.py:142-143); stop-gradient on the features of the copy (head.py:196) | missing: three inputs, several outputs, stop-gradient, depthwise |
| `DFL` | identity at `reg_max = 1` (head.py:139; yaml:10) | not needed |

Existing layers that YOLO26 uses unchanged: `Conv2d`, `MaxPool2d`, `Upsample2d`. Everything else is new.

**Attention in NHWC (derived here from block.py:1320-1334).** No permute is needed.

- The `qkv` output is `[B, H, W, heads * (2*kd + hd)]`, head-major on the channel axis (block.py:1323-1325). Per head, three `SLICE_CHANNELS` give q and k as `[B,H,W,kd]` and v as `[B,H,W,hd]`.
- `RESHAPE` to `[B,N,kd]` and `[B,N,hd]` gives the transposes qᵀ, kᵀ and vᵀ of the reference tensors.
- `attn = softmax_last(scale * (qᵀ · (kᵀ)ᵀ))` is one rank-3 matmul with `transB`, of shape `[B,N,N]`.
- `(v @ attnᵀ)ᵀ = attn · vᵀ` is a second rank-3 matmul, of shape `[B,N,hd]`.
- Reshape to `[B,H,W,hd]`, then `CONCAT_CHANNELS` over heads.

The batch axis of each matmul is B, and there are two matmuls per head. Stacking heads on the batch axis would need a non-channel concat or a permute, which SushiAI does not have (bc:777-779). `num_heads = max(c // 64, 1)`, `head_dim = dim // num_heads`, `key_dim = int(head_dim * 0.5)` (block.py:1099, 1301-1303).

**BatchNorm, train mode.** `BATCHNORM2D` outputs `y`, `mean` and `inv_std` (bc:909; low:1258-1260). Its variance is a lowering scratch (low:1273): the biased estimate (`scal(inverse_rows, variance)`, low:1288-1289), with epsilon added in place (low:1292) and then overwritten as `1/sqrt` (low:1294-1296). The running-mean update is `scal` and `axpy` on the `mean` output. The running-variance update needs the batch variance before epsilon, which no output carries. It needs one of:

- the variance as a fourth output of `BATCHNORM2D`;
- the update recorded inside the BN lowering, where the scratch exists, with the two running buffers as operands of the node.

Recovering it as `1 / inv_std^2 - eps` loses precision when the variance is small against `eps = 1e-3`. The choice changes the node's interface (decision D13). Whether the reference scales the running variance by n/(n-1) is in section 9.

**Initialisation from scratch.** `apply_initializer` zeroes a parameter when it is a bias or belongs to the final layer, and gives every other parameter a fan-scaled random fill (src/nn/init.cpp:100-119). `ParameterRole` has `WEIGHT` and `BIAS` only (include/SushiAI/nn/module.hpp:51-57). `InitPolicy::zero_final_layer` defaults to true and zeroes every parameter from the last weight in walk order onward (include/SushiAI/nn/init.hpp:140, 183-189). Training YOLO26 from scratch needs:

- a ones initialiser for BN gamma, which today is random as `WEIGHT` or zero as `BIAS`;
- a constant initialiser for the head biases (head.py:229-241);
- a per-parameter initialiser rule in place of the role and final-layer test;
- conv weights at torch's default, because `initialize_weights` runs after `bias_init` (U/nn/tasks.py:520, 525) and does `pass` for `nn.Conv2d` (torch_utils.py:678-679). It sets only BN `eps = 1e-3` and `momentum = 0.03` (torch_utils.py:681-682).

`init.cpp` calls `engine.random()` (init.cpp:52, 62, 88), so these wait for the migration.

## 3. Model topology

The module seam is one `ValueId` in and one out (include/SushiAI/nn/module.hpp:216-225, include/SushiAI/nn/ilayer.hpp:57-58), `LayerFactory` takes one input `Shape` (include/SushiAI/nn/layer_registry.hpp:51-53), and `build_graph` stores one output (include/SushiAI/train/trainer.hpp:148).

YOLO26 needs (U/cfg/models/26/yolo26.yaml:22-52):

- **References to earlier outputs.** Rows 12, 15, 18 and 21 concat with rows 6, 4, 13 and 10. Row 23 reads rows 16, 19 and 22.
- **Multi-input nodes.** `Concat` and `Detect`.
- **A multi-output model.** Training returns boxes and scores for 3 levels and 2 heads, 12 values in total.

**C++ layer API.** A hand-written `Module` can already emit skips and concats inside its own `forward`, because it receives the `GraphBuilder`. That covers Bottleneck, C2f, C3k, SPPF, PSABlock and C2PSA as one-in one-out blocks. The backbone-to-neck routing and `Detect` do not fit.

**Model-graph container.** The routing needs a container that holds a list of nodes, each with a `from` list, a set of saved outputs, and a forward that returns several values. For the container to hold `Concat` and `Detect` without testing a node's kind, every node presents one seam:

- forward takes `span<const ValueId>` and returns several values;
- the factory takes `span<const Shape>`;
- one adapter wraps any existing `ILayer` as a one-in one-out node;
- `Concat` and `Detect` implement the seam directly.

The seam is a public interface (decision D5).

**JSON model file.** The registry has 13 single-input rows (layer_registry.hpp:182-196). It needs:

- a `from` field per row, with Ultralytics' convention of -1 or an absolute index;
- rows for the composite blocks;
- an `outputs` list.

Depth and width scaling (`parse_model`) can live in a generator, or the n-scale channels can be written literally.

**Parameter naming.** A type that offers `walk_parameters(prefix, visitor)` names its own subtree (module.hpp:249-255, 306-311). Index naming (`0.weight`) is the fallback for `visit_children` containers (module.hpp:312-318). A composite block can therefore emit paths such as `m.0.cv1.conv.weight` today. `fields()` lists `Parameter` members only (module.hpp:131-138), so each composite writes that walk by hand. Mirroring Ultralytics' paths, for example `model.22.m.0.1.attn.qkv.conv.weight` (block.py:1096-1100), makes the importer's name mapping the identity.

**Buffers.** BN running mean and variance are non-trainable state. `ParameterRole` has only `WEIGHT` and `BIAS` (module.hpp:51-57), and nothing walks a non-parameter tensor.

**Checkpoints.** `save_checkpoint` and `load_checkpoint` walk `for_each_parameter` only (include/SushiAI/io/checkpoint.hpp:69-82). Load rejects a record the model lacks (checkpoint.hpp:157-159) and a parameter the file lacks (checkpoint.hpp:180-183). A record is a name, a dtype and a shape (checkpoint.hpp:90-99), so nested names fit. The reader accepts only SushiAI's `SACP` format (src/io/checkpoint_format.cpp:169-174). BN buffers, the EMA set and the optimizer state (two momentum buffers per muon parameter) cannot be saved or loaded without changing those two functions, and the header includes `SushiBLAS/engine.hpp` (checkpoint.hpp:36).

**Weight importer.** The importer exists for the parity checks of stages b and c. Its job:

- **Source.** An Ultralytics `.pt` is a pickled `nn.Module` in half precision (`deepcopy(ema).half()`, U/engine/trainer.py:785; `torch.save` of a dict with `"model": None, "ema": ema`, trainer.py:796-802). C++ cannot read it. A Python exporter reads the module, or builds a model from the yaml, and writes tensors to a file SushiAI reads. Weights taken from a `.pt` are fp16-rounded, and the reference forward must run after `.float()`.
- **Conv weight.** `(O, I, kh, kw)` becomes `[C_out, kh, kw, C_in]` by the permutation (0, 2, 3, 1) (SPEC:30).
- **Depthwise weight.** `(C, 1, kh, kw)` becomes tap-major, channel-minor, `[kh, kw, C]`, which is the order the composition of section 1 and `im2col` use (bc:557). The stored shape is fixed when the op is designed.
- **Per-channel vectors.** Conv bias, BN weight, BN bias, BN `running_mean` and BN `running_var` are `[C]` and are copied unchanged. The two running vectors go to buffers.
- **Names.** Each Ultralytics state-dict key maps to one SushiAI path. An entry with no counterpart is an error unless a list names it as ignored.
- **One-to-one head.** It is a `deepcopy` of the one-to-many head at construction (head.py:142-143) and has its own entries under `one2one_cv2` and `one2one_cv3`.
- **Reference tensors.** A fixed input goes from NCHW to NHWC by (0, 2, 3, 1). Reference boxes `[B, 4, A]` and scores `[B, nc, A]` concatenate the levels on the anchor axis (head.py:176); anchor `h*W + w` of a level is element `[b, h, w, :]` of SushiAI's per-level output. A conv-weight gradient takes the conv-weight permutation.

Where the exporter runs and which file it writes are decision D12.

## 4. Training loop

| Item | Reference | Where it runs | Missing today |
| --- | --- | --- | --- |
| Loss per branch: CIoU box (gain 7.5), BCE cls (0.5), L1 on ltrb (1.5), each divided by `target_scores_sum`, the 3-vector times batch size | loss.py:132-157, 448-498; gains U/cfg/default.yaml:103, 104, 106 | compiled path, as a fixed graph over full-shape leaves | ops of section 1; the general loss root (diff:142-154); `NLL_LOSS` cannot be a term of a sum (loss.cpp:50-53); the objective registry has one row (include/SushiAI/train/objective.hpp:239-241) |
| Branch weights: `loss_o2m * o2m + loss_o2o * o2o`, summed to a scalar by the trainer | loss.py:1344; trainer.py:547 | compiled path | as above |
| `o2m` from 0.8 to 0.1, linear over epochs, `o2o = 1 - o2m`, updated once per epoch | loss.py:1331-1354; trainer.py:638-639, and on resume 1139-1140 | host; fed as two 1-element leaves | a per-step leaf feed beyond `inputs` and `targets` (src/train/trainer.cpp:259-260) |
| Label assignment, twice (top-k 10; top-k 7 then 1) | loss.py:1327-1328, 439-446; tal.py:63-386 | host, on detached predictions | see "Label assignment by a second forward pass" below and D2 |
| MuSGD | U/optim/muon.py:38-49, 89-127, 235-265; trainer.py:1180-1232 | device update | see "MuSGD" below |
| AdamW when `optimizer: auto` and iterations <= 10000 | trainer.py:1180; default.yaml:27 | exists | none for a run above 10000 iterations |
| Schedule: linear from 1 to `lrf`, stepped per epoch | trainer.py:268-274; default.yaml:95 | host, through mutable `options()` (include/SushiAI/optim/sgd.hpp:95-96) | scheduler object; per-group lr |
| Warmup: `nw = round(warmup_epochs * nb)` iterations; per-iteration lr, momentum and accumulate ramp | trainer.py:276-279, 518-535 | host | per-group lr. Momentum is already rewritten before every replay (src/optim/sgd.cpp:111; src/optim/optimizer.cpp:88-91; src/graph/compiled_step.cpp:66), but must be non-zero at construction (sgd.cpp:98-102) |
| Gradient accumulation to `nbs = 64` | trainer.py:321, 598-600; default.yaml:114 | compiled path plus host | a summed gradient buffer (the slot's gradient is "rewritten by every step", include/SushiAI/optim/optimizer.hpp:85) and a call to `step()` every k-th replay. `OptimizerPlacement::PER_STEP` already steps on the host after the replay (trainer.hpp:228-245; trainer.cpp:272) |
| Gradient clip, global L2 norm 10.0 | trainer.py:905-911 | device norms, host factor | SushiAI code that combines per-tensor `nrm2` values (red:186) into one factor. The SGD kernel already multiplies every gradient by a per-step scalar (sgd.cpp:155, 177), and `PER_STEP` gives a host point between backward and update |
| EMA: `d = 0.9999 * (1 - exp(-updates / 2000))`, `ema += (1 - d) * (model - ema)` over every floating state-dict entry, weights and BN buffers | torch_utils.py:838, 852, 868-880 | device, after each optimizer step | whole feature; needs BN buffers |
| Mixed precision | default.yaml:36 | optional; fp32 is a supported reference path | a loss scaler exists (`include/SushiAI/optim/loss_scaler.hpp`); not needed for a first run |
| Validation: EMA model, eval BN, `max_det` 300, mAP50-95 | U/engine/validator.py; U/utils/metrics.py; default.yaml:55, 57 | forward on device; decode, NMS or top-k, matching and AP on host | eval mode; host evaluator; rectangular batches (below); `IObjective::score` anticipates mAP only in a comment (objective.hpp:144) |

### The loss as a fixed graph

Derived here from loss.py:132-157 and 420-498. Per level and per branch the host assigner writes leaves at the full shape of what they multiply, and the graph sums over all anchors. Predicted boxes and distances are in grid units; `target_bboxes` is divided by the stride before the box loss (loss.py:461).

- **Common scalars.** `weight = target_scores.sum(-1)` per anchor, zero off the foreground (loss.py:133). `tss = max(target_scores.sum(), 1)` (loss.py:448).
- **Class term.** BCE with logits over every anchor and class, no mask, summed and divided by `tss` (loss.py:451-454). Leaves: soft targets `[B,H,W,nc]`.
- **CIoU term.** `sum((1 - ciou) * weight) / tss` over foreground anchors (loss.py:134-135). Leaves: target boxes `[B,H,W,4]`, and `weight / tss` at the shape it multiplies.
- **L1 term.** `target_ltrb = bbox2dist(anchor_points, target_bboxes)` without a clamp, because `reg_max` is not passed (loss.py:143; tal.py:479-494). Both sides are multiplied by the stride; columns 0 and 2 are divided by the image width and columns 1 and 3 by the image height; the absolute differences are averaged over the four sides; the mean is multiplied by `weight`, summed and divided by `tss` (loss.py:144-155). `imgsz` is the first feature map's height and width times the first stride (loss.py:428), so it follows the batch's input size. Every factor is non-negative, so the term equals `sum(abs(pred_dist - target_ltrb) * K)` with one leaf `K[B,H,W,4] = (stride / image side of that column) * (1/4) * weight / tss`. Leaves: `target_ltrb` and `K`.
- **Gains and batch size.** The three terms are scaled by 7.5, 0.5 and 1.5 (loss.py:469-471) and the vector by the batch size (loss.py:498). Both fold into `SCALE`.

The class term matches the reference exactly, because the reference does not mask it. The CIoU and L1 sums equal the gathered sums when off-foreground rows contribute zero.

**Hazard: off-foreground CIoU must stay finite.** `bbox_iou` guards with `eps` in `h1`, `h2`, `union`, `c2` and `alpha` (metrics.py:145-146, 154, 162, 169). The reference evaluates foreground rows only (loss.py:132-134). Over all anchors, an off-foreground predicted box is unconstrained: a non-positive `h1` or `union` gives an infinite or NaN ratio, and a NaN times a zero weight is NaN in the sum and in the gradient. A dummy target box does not bound the predicted side. The remedy derived here is to blend the prediction before CIoU, `pred' = m * pred + (1 - m) * dummy` with a `[B,H,W,4]` mask leaf `m` and a well-formed dummy box, using existing `MUL` and `ADD`. Off-foreground rows then evaluate CIoU on the dummy box against itself, and the gradient into `pred` is multiplied by zero. This is untested (section 9).

### Label assignment by a second forward pass

The reference assigns from detached predictions of the same forward (loss.py:439-446). In SushiAI the compiled step is one replay (trainer.cpp:264-265), `add_host` and `add_emitter` appear nowhere under `include/` or `src/`, and host reads happen after the plan through `data_as` (trainer.cpp:199-204, 226-231). The eager front end excludes backward across a mid-step read in its first version (docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:188-190). Assigning on the host between two replays (D2 option 1) needs these units:

1. **A forward-only step.** A second graph, compiled into a second `CompiledStep`. A `TensorPool` adopts one plan (low:48-52), so the second step has its own pool.
2. **Shared parameters.** The second pool's parameter leaves are bound to the first pool's tensors with `TensorPool::bind` (low:65-70; include/SushiAI/graph/lowering.hpp:101). Whether two compiled steps can share storage this way has not been read further (section 9).
3. **No statistics update in the forward-only graph.** It runs BN on batch statistics, as the training forward does, and records no running-statistics update, or every step would update twice.
4. **Objective-declared observed values.** The 12 predictions must be pinned and observed. `build_graph` observes `inputs`, `targets`, `logits` and `loss` (trainer.hpp:176-178), and other activations are packed into arenas (src/graph/memory_plan.cpp:360).
5. **A host stage in `Trainer`.** `Trainer` builds one `step_` from one graph (trainer.cpp:180-182). It needs to run the forward-only step, read the 12 values whole, call the objective's host stage, feed every declared leaf, and then run the full step.

The cost is one extra forward per training step.

### MuSGD

As the code has it (muon.py:38-49, 89-127, 235-265):

- **Muon momentum.** `buf = beta * buf + (1 - beta) * g`; with Nesterov, `u = beta * buf + (1 - beta) * g` (muon.py:91-95). `beta` is the group's `momentum` (muon.py:249).
- **Flatten.** A rank-4 update is flattened to `(O, I*kh*kw)` (muon.py:105). A permutation of the columns permutes the result identically (muon.py:100-102), so SushiAI's `[C_out, kh*kw*C_in]` reshape gives the same update in its own layout.
- **Newton-Schulz.** `X /= norm_F(X) + eps` with `eps = 1e-7` (muon.py:10, 40); transposed when rows > columns; five iterations of `A = X Xᵀ`, `B = b*A + c*A*A`, `X = a*X + B*X` with `a, b, c = 3.4445, -4.7750, 2.0315` (muon.py:41-49). The reference computes in bfloat16 (muon.py:39).
- **Scale.** Each update is multiplied by `max(1, rows / cols) ** 0.5`, where rows and cols are the last two sizes of the gradient in the reference layout (muon.py:106). For `(O, I, kh, kw)` that is `kh / kw`, which is 1 for every square kernel. SushiAI must compute it from the reference layout, not from `[C_out, kh, kw, C_in]`.
- **Muon step.** `p -= lr * 0.2 * u` (muon.py:252; trainer.py:1216).
- **SGD half, same parameters.** After the muon step: `g' = g + wd * p` with the already-updated `p` (muon.py:259-260), `buf2 = momentum * buf2 + g'`, `u2 = g' + momentum * buf2` with Nesterov, `p -= lr * 1.0 * u2` (muon.py:254, 261-264). Weight decay enters this half only. This matches SushiAI's SGD kernel with dampening 0 (sgd.cpp:176-191).
- **Batching.** The reference zero-pads matrices with equal column count into one batched call (muon.py:98-126). Zero rows stay zero through the iteration (muon.py:119-120) and add nothing to a matrix's norm, so one call per parameter with whole-tensor `nrm2` and rank-2 `gemm` computes the same update.

**Parameter groups** (trainer.py:1186-1232):

| Group | Members | Weight decay | Update |
| --- | --- | --- | --- |
| muon | every parameter of rank 2 or 4, which includes depthwise weights `(C, 1, 3, 3)` | scaled decay | muon step, then SGD half |
| bias | names containing `bias` | none set (the group is built without a `weight_decay` entry, trainer.py:1213) | Nesterov SGD |
| bn | parameters of normalisation modules | 0.0 | Nesterov SGD |
| weight | the rest | scaled decay | Nesterov SGD |

- **Scaled decay.** `weight_decay * batch_size * accumulate / nbs` (trainer.py:322).
- **Tripled learning rate.** Under MuSGD only, every parameter of `cv3` and `one2one_cv3`, in every group, gets `lr * 3` (trainer.py:1217-1231). The comment says "when finetuning"; the code applies it unconditionally.
- **Values under `optimizer: auto`.** The optimizer is built with lr 0.01 and momentum 0.9, and `warmup_bias_lr` is set to 0.0 (trainer.py:1180-1182). An explicitly named `MuSGD` keeps `momentum: 0.937` and `warmup_bias_lr: 0.1` (default.yaml:96, 100). Warmup writes each group's momentum as an interpolation from `warmup_momentum` 0.8 to `args.momentum` (trainer.py:534-535), and `args.momentum` is referenced only there and at trainer.py:329, so an `auto` run leaves warmup near 0.937, not 0.9. The same value is the muon `beta` (muon.py:249).

SushiAI needs: a second momentum buffer per muon parameter, a new update beside `SGD_UPDATE`, and per-parameter rank, role and owner path for grouping. `ParameterSlot` holds a name, a parameter and a gradient (optimizer.hpp:77-87), and groups change it and `bind_parameters` (trainer.cpp:38-85).

### What each type must carry that it cannot today

- **`Batch`** (include/SushiAI/data/dataset.hpp:50-67): a rank-4 image tensor and a variable-length box list (image index, class, xywh) kept on the host. Today it holds `[N,D]` inputs, `[N,C]` one-hot targets and `int32` labels, as `SushiBLAS::Tensor` fields.
- **`Dataset`**: no `num_classes`-equals-logits contract; a host-side box accessor; an epoch hook for `close_mosaic` (trainer.py:505; default.yaml:34). The factory takes `SushiBLAS::Engine&` (dataset.hpp:272-274).
- **`IObjective`**: several model outputs (`loss()` takes one `ValueId`, objective.hpp:138-141); declared observed values; a host stage that reads predictions and fills declared leaves each step; a per-epoch update.
- **`Trainer`**:
  - a rank-4 input leaf (trainer.hpp:143 fixes `Shape{batch, features}`);
  - no required `targets` and `logits` pair (trainer.cpp:184-186);
  - no class count read from the logits (trainer.cpp:188-190) or checked against the dataset (trainer.cpp:283-285);
  - feed every declared leaf, not two (trainer.cpp:259-260);
  - whole-tensor host reads of 12 outputs;
  - a second, forward-only step and the host stage between the two;
  - scheduler, warmup, accumulation, clipping, EMA and eval mode.

## 5. Data

**Format** (U/data/utils.py:141-143, 440-491; not reopened for this map): images under `.../images/...`, labels at the same path with `/labels/` and `.txt`. Each row is `class xc yc w h`, normalised. An empty or missing file is a background image.

**Training pipeline.** Mosaic, random affine, HSV, horizontal flip, format (U/data/augment.py:2870-2912; not reopened), then `/255` (U/models/yolo/detect/train.py:119), RGB, no mean or std.

**Validation pipeline.** `LetterBox(new_shape=(imgsz, imgsz), scaleup=False)` (U/data/dataset.py:339), in rectangular batches: the validation dataset is built with `rect=True` (detect/train.py:76). Images are sorted by aspect ratio, and each batch gets its own shape, `ceil(shape * imgsz / stride + pad) * stride` with the long side at `imgsz` (U/data/base.py:405-427). The input height and width therefore vary between validation batches. The validation batch size is twice the training batch size (trainer.py:314-319). SushiAI compiles one input shape per step, so it either compiles one step per distinct batch shape or validates at a fixed square letterbox, which departs from the reference (decision D7).

The pipeline runs on the host over `uint8`. No SushiBLAS kernel is needed. JPEG decoding needs a library (decision D6).

| Piece | Default | Needed for a faithful run | Cost of leaving it out of a first run |
| --- | --- | --- | --- |
| Label parser, resize, letterbox, `/255` | on | yes | cannot train or validate |
| Rectangular validation batches | on for validation | yes | validation numbers differ from the reference by an unmeasured amount |
| Horizontal flip p=0.5 (default.yaml:124) | on | yes | not quantified in any source read |
| HSV jitter 0.015, 0.7, 0.4 (default.yaml:115-117) | on | yes | lower accuracy, not quantified; parity needs OpenCV's 8-bit HSV convention |
| Random affine, translate 0.1, scale 0.5 (default.yaml:119-120), with box filter | on | yes | lower accuracy, not quantified |
| Mosaic p=1.0, off for the last 10 epochs (default.yaml:126, 34) | on | yes | the largest departure from the recipe; cost not quantified. Overfit and loss-parity stages do not need it. |
| MixUp, CutMix, CopyPaste, rotation, shear, perspective, vertical flip | off | no | none |
| Albumentations (4 effects at p=0.01) | only if installed | no | none against an environment without the package |
| Multi-scale | off | no | none |

## 6. What SushiBLAS must provide

This section is written to be read on its own.

**Paths.** `sushiblas:` is `D:\Projects\sushiblas\include\SushiBLAS`. `sushiai:` is `D:\Projects\sushiai`. `U/` is `ultralytics/` at commit `ce1399439b93c490b648b55dedd0a6334f0199da`.

**Conventions for every item.**

- Element type FLOAT32. Index outputs INT64 where a kernel has them today. No item needs HALF, FLOAT64, a boolean dtype, int to float conversion or broadcasting between unequal shapes.
- Rank-4 tensors are NHWC, row-major.
- Operands are contiguous unless an item says otherwise. SushiAI's lowerings produce contiguous values: `RESHAPE` and `BATCHNORM2D` reject a strided operand (sushiai: src/graph/lowering.cpp:938, 1251).
- A backward is a fixed-order gather per output element, never an atomic scatter (sushiblas: engine/README.md:236-243).
- Names are today's. Under v0.1.0 the names, return types and view construction may change; the semantics stated here may not.

**Result.** YOLO26's forward pass and its training need no new SushiBLAS kernel. They need the existing kernels below kept with the stated semantics. Two new kernels are requests under owner decisions, for memory and speed. Nothing has executed, so each "exists" is a declaration read in a header.

### 6.1 Required for the forward pass

| # | Operation (today's declaration) | Operands and semantics | Views | Used by |
| --- | --- | --- | --- | --- |
| F1 | `im2col` with stated extents (sushiblas: engine/spatial.hpp:59) | `in [N,H,W,C]` to `cols [N*Ho*Wo, R*S*C]`; column index `(r*S + s)*C + c`, channel innermost; out-of-bounds taps are zero; padding is a parameter (sushiblas: engine/README.md:246-248). Geometries: 3x3 stride 1 pad 1, 3x3 stride 2 pad 1 | contiguous input suffices | every 3x3 `Conv`; the depthwise composition (F4) |
| F2 | `gemm`, rank 2 (sushiblas: engine/blas/level3.hpp:51) | `C = alpha * op(A) * op(B) + beta * C`, both transpose flags | contiguous; today's kernel rejects a transposed view or a slice (sushiblas: engine/blas/README.md:270-274) | every convolution: `cols [rows, taps]` times the `[C_out, taps]` filter transposed |
| F3 | `gemm`, rank 3, batched (same declaration) | the output sets the batch count; an input with the same batch count is stepped matrix by matrix (sushiblas: engine/blas/README.md:276-279). Two shapes: `A [B,N,kd]`, `B [B,N,kd]` with `transB`, `C [B,N,N]`; and `A [B,N,N]`, `B [B,N,hd]`, `C [B,N,hd]` | contiguous | attention, two calls per head (U/nn/modules/block.py:1331-1332). N = H*W of the stride-32 map, 400 at a 640 px input |
| F4 | `channel_affine` (sushiblas: engine/spatial.hpp:159) | `out[n,h,w,c] = gamma[c] * in[n,h,w,c] + beta[c]`; `in`, `out` `[N,H,W,C]`, `gamma`, `beta` `[C]` | contiguous | BN in eval form; BN centring; the depthwise composition, where C is 9 times the layer's channel count |
| F5 | `concat_channels`, `slice_channels` (sushiblas: engine/spatial.hpp:140, 149) | rank 4, channel axis; a slice copies `width` channels from a channel offset into its own tensor | contiguous | Concat rows, C2f, C3k, SPPF, C2PSA, attention q/k/v, depthwise composition |
| F6 | `pool_max` (sushiblas: engine/spatial.hpp:86) | 5x5, stride 1, pad 2; `indices` INT64 shaped like `out`; the running maximum starts at the lowest finite value, so padding is never a candidate (sushiblas: engine/README.md:276-283) | contiguous | SPPF, three in series (U/nn/modules/block.py:229-240) |
| F7 | `upsample_nearest` (sushiblas: engine/spatial.hpp:122) | factor 2 on H and W | contiguous | rows 11 and 14 of the model |
| F8 | `silu`, `sigmoid`, in place (sushiblas: engine/math/nonlinear.hpp:143, 92) | `x * sigmoid(x)`; `sigmoid(x)` | contiguous | every `Conv` with activation; class scores at inference |
| F9 | `add`, `sub`, `mul`, same shape (sushiblas: engine/math/elementwise.hpp:60, 70, 80) | `C = A op B`, operands of rank 1 to 4 with equal shapes. Whether today's kernels accept rank-4 operands was not read in the kernel bodies | contiguous | shortcuts, box decode, BN |
| F10 | `copy`, `scal`, `axpy` (sushiblas: engine/blas/level1.hpp:70, 65, 53); `gemv`, `ger` (sushiblas: engine/blas/level2.hpp:52, 58) | as today; `scal` and `axpy` take a `double` scalar; `ger` adds an outer product into a rank-2 matrix; `gemv` has a transpose flag | contiguous | reshape as a copy, `q * scale`, bias add (`copy` + `ger`), softmax (F11) |
| F11 | Row softmax pieces: `max_along` (rank 2, axis 1), `ger`, `exp`, `gemv`, `log` (sushiblas: engine/math/reductions.hpp:152; engine/math/elementwise.hpp:130, 138) | SushiAI composes `log_softmax` over the last axis of a `[R, N]` matrix from these, max-shifted, then `exp` (sushiai: src/graph/lowering.cpp:838-852). R = B*N, N as in F3 | contiguous | attention softmax (U/nn/modules/block.py:1331) |
| F12 | `sum_along` (rank 2, axis 0), `sqrt`, `reciprocal` (sushiblas: engine/math/reductions.hpp:161; engine/math/elementwise.hpp:98, 114) | per-channel sums over a `[N*H*W, C]` matrix; in-place root and reciprocal on `[C]` | contiguous | BN (sushiai: src/graph/lowering.cpp:1278-1296) |
| F13 | Reinterpretation of contiguous storage under another shape with the same element count, without a copy | today a `Tensor` constructed over the same storage, offset and layout with new extents (sushiai: src/graph/lowering.cpp:945-949, 1220-1221) | this is the view | BN reads `[N,H,W,C]` as `[N*H*W, C]`; reshape |

### 6.2 Required for training, in addition

| # | Operation (today's declaration) | Operands, semantics and backward formula | Used by |
| --- | --- | --- | --- |
| T1 | `col2im` (sushiblas: engine/spatial.hpp:77) | adjoint of F1 as a gather: each image element sums its column entries in a fixed order | backward of every 3x3 convolution |
| T2 | `pool_max_backward`, `upsample_nearest_backward` (sushiblas: engine/spatial.hpp:96, 131) | as today | SPPF, upsample rows |
| T3 | `silu_backward(dy, x, dx)` (sushiblas: engine/math/nonlinear.hpp:153) | `dx = dy * s * (1 + x * (1 - s))`, `s = sigmoid(x)`, `x` the forward input | every `Conv` with activation |
| T4 | `softplus` in place, `softplus_backward(dy, x, dx)` (sushiblas: engine/math/nonlinear.hpp:179, 189) | `log(1 + exp(x))`; `dx = dy * sigmoid(x)` | BCE with logits, as `softplus(x) - x*t` (U/utils/loss.py:451-454) |
| T5 | `relu`, `relu_backward` (sushiblas: engine/math/nonlinear.hpp) | as today | `clamp_(0)` in CIoU (U/utils/metrics.py:149-151) |
| T6 | `div`, `min`, `max`, same shape (sushiblas: engine/math/elementwise.hpp:90, 276, 286); `abs`, `atan`, in place (:146, :202) | forward only, as today, on `[B,H,W,1]` and `[B,H,W,4]` operands | CIoU (U/utils/metrics.py:143-170); L1 (U/utils/loss.py:152-154) |
| T7 | The caller-functor seams `Dispatch::execute_unary_inplace`, `execute_binary`, `execute_zip_inplace` | a caller-supplied elementwise functor over one in-place tensor, over two inputs and one output, and over k equally shaped tensors updated in place. SushiAI writes these bodies on them: `div` backward `da = dc / b`, `db = -dc * a / b^2`; `min` backward `da = dc * [a <= b]`, `db = dc * [a > b]`, and `max` with the comparisons reversed; `abs` backward `dx = dy * sign(x)`; `atan` backward `dx = dy / (1 + x^2)`; the MuSGD and EMA updates. SushiBLAS may offer any of them as a named kernel instead; none is required as one | CIoU and L1 backward; optimizer; EMA. Today's users: sushiai: src/graph/lowering.cpp:738-744, src/optim/sgd.cpp:168 |
| T8 | A functor that reads a 1-element tensor's storage at execution time | today the functor captures the factor's data pointer, so the host can rewrite the value between replays without recompiling (sushiai: src/graph/lowering.cpp:733-744). The storage address must stay fixed across replays | loss-scale and clip factor on every gradient |
| T9 | `nrm2` over a whole tensor, to one element (sushiblas: engine/math/reductions.hpp:186) | as today | gradient clipping (one norm per gradient tensor, combined on the host); the Frobenius norm of one matrix in Newton-Schulz. Loss totals use `gemv` (F10), not a new reduction |
| T10 | `gemm`, rank 2, with `alpha` and `beta` other than (1, 0) | Newton-Schulz on one `[m, n]` matrix, five iterations: `A = X * Xᵀ` (`transB`); `B = c * A * A + b * A` (copy A into B, then `alpha = c`, `beta = b`); `X = B * X + a * X` (`alpha = 1`, `beta = a`), with a second buffer for X. When m > n the same recurrence runs on `XᵀX` through `transA`. `a, b, c = 3.4445, -4.7750, 2.0315`. Before the loop, `X /= nrm2(X) + 1e-7` (U/optim/muon.py:10, 40-49) | MuSGD, once per rank-4 parameter per step |
| T11 | `scal`, `axpy` on per-channel and whole-parameter tensors (sushiblas: engine/blas/level1.hpp:53, 65) | `r = (1 - m) * r + m * stat`; `ema = d * ema + (1 - d) * w` | BN running statistics; EMA (U/utils/torch_utils.py:852, 880) |

The batched, zero-padded Newton-Schulz of the reference (U/optim/muon.py:98-126) is not needed: zero rows stay zero, so one call per parameter with whole-tensor `nrm2` gives the same update. A per-matrix norm over a rank-3 batch is not requested.

### 6.3 Only under an owner decision

**O1. Depthwise convolution triple (decision D4).** Without it SushiAI composes depthwise from F1, F4, F5 and F9 at the cost of a `[N, Ho, Wo, 9C]` column tensor and an affine result of the same size per layer. Every grouped convolution in YOLO26 has groups equal to channels (U/nn/modules/head.py:132-133; U/nn/modules/block.py:1309; U/nn/modules/conv.py:209), so a general group count is not requested. A grouped `im2col` feeding a batched `gemm` over groups is not requested either: with today's channel-innermost column order it needs a permute on each side.

- Forward: `x [N,H,W,C]`, `w [kh,kw,C]`, a `ConvGeometry`, `y [N,Ho,Wo,C]`.
  `y[n,i,j,c] = sum over r,s of x[n, i*sh - pt + r*dh, j*sw - pl + s*dw, c] * w[r,s,c]`, out-of-bounds taps zero.
- Backward for the input: `dx[n,h,w,c] = sum over the (r,s,i,j) with i*sh - pt + r*dh = h and j*sw - pl + s*dw = w of dy[n,i,j,c] * w[r,s,c]`, gathered per element of `dx`.
- Backward for the weight: `dw[r,s,c] = sum over n,i,j of dy[n,i,j,c] * x[n, i*sh - pt + r*dh, j*sw - pl + s*dw, c]`, reduced in a fixed order per element of `dw`.
- YOLO26 uses 3x3, stride 1, pad 1, dilation 1, no bias. C is 64, 80, 128 or 256 at the n scale with 80 classes; the 80 is `max(ch[0], min(nc, 100))` (U/nn/modules/head.py:123) and follows the class count.
- Contiguous operands suffice.

**O2. Row softmax pair (decision D4).** Without it the forward is the nine-task composition of F11 plus `exp`, and SushiAI lowers the backward from `mul`, `gemv` and `ger`.

- `softmax(x, y)`: `x`, `y` `[R, N]`, over the last axis, max-shifted.
- `softmax_backward(dy, y, dx)`: `dx = y * (dy - rowsum(dy * y))`, with the row sum broadcast over the N columns.
- R = B*N and N = 400 at a 640 px input. Contiguous operands suffice.

**O3. Device-side label assignment (decision D2, option 3).** Non-differentiable, FLOAT32 values with INT64 indices: top-k along the last axis of rank 3, returning values and indices; scatter and gather along an axis; argmax and max along an axis of rank 3; tensor-scalar comparison. Host assignment removes this item.

**O4. Gather-based loss (decision D3, if the fixed-shape form is rejected).** Broadcasting binaries, concat along a non-channel axis, and a row gather whose backward is a scatter-add. The scatter-add has to be expressed as a gather to keep the determinism rule above.

### 6.4 Not requested

Softmax on arbitrary axes, general permute, a boolean dtype, int to float conversion, bilinear resize, `atan2`, tensor-tensor `pow`, `randperm`, a `sign` kernel, a grouped `im2col`, a per-matrix norm over a batch. Under D1, D2 option 1 or 2, D3 and host-side data, YOLO26 does not need them.

### 6.5 For the runtime, not SushiBLAS

A host task inside one plan with declared reads and writes (`add_host(Reads, Writes, fn)`, `D:\Projects\sushiruntime\include\SushiRuntime\api\graph\README.md:375-382`) is what would let host label assignment sit between forward and loss in a single replay (D2 option 2).

## 7. Stages

"Now" means host-lane code: IR ops, shape rules, gradient rules, layer classes and host algorithms, with no engine call. "After migration" means the unit touches a file that calls the engine. SPEC:27 and SPEC:33 record those files as frozen until the SushiBLAS migration.

**Lowerings are written once, after the split.** `Lowering::record_node` is one `if (node.op == Ops::X.id)` chain (low:389-429) with inner chains at low:695-697 and low:757-761, and its fallthrough message reads "Add it to Lowering::record_node" (low:441). Gradient rules are already per-domain tables (all.hpp:51-57), and the estate audit recommends the same shape for lowering (docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md:902-904; docs/design/REMAINING_WORK.md:104-107). SushiAI's migration to SushiBLAS v0.1.0 splits `lowering.cpp` into per-domain tables. The new lowerings of this map (`SILU` and its backward, `DIV`, `MIN`, `MAX`, `ATAN`, `ABS`, `SOFTPLUS`, their backwards, the softmax backward, `STOP_GRADIENT`, rank-3 `MATMUL`, the BN statistics update) are each written once, after that split, into the table of their domain. None is added to the chain.

| Stage | Check | Units | Repository | Timing |
| --- | --- | --- | --- | --- |
| 0. CV close | the existing MNIST CNN gate trains (SPEC:13-14) | owner's build of the current tree | sushiai | owner's build; everything below depends on it |
| a. YOLOv3-tiny-class body forward | output shape, and values against a host reference on fixed weights | model-graph container with the node seam of section 3 (D5); `Concat` node; `ILayer` adapter; rank-4 input leaf; `BatchNorm2d` in eval form (`CHANNEL_AFFINE` over folded buffers); buffers; JSON `from` and `outputs`; named children | sushiai | container, seam, layers, JSON and naming: now. Rank-4 leaf (`build_graph`, `Trainer`, `Dataset`), buffers in the checkpoint, the eval fold: after migration. |
| r. Reference files | the exporter's own forward reproduces Ultralytics' output on the fixed input | Python exporter at the pinned commit (D11): weights, BN buffers, a fixed input, the 12 head outputs, six loss items (three per branch, from `criterion.one2many.loss` and `criterion.one2one.loss`, since `E2ELoss.__call__` returns only the one-to-one items, loss.py:1344), one conv-weight gradient, and `running_mean` and `running_var` of one BN after one step; all in fp32, after `.float()` | outside the repositories, or under `tools/` | needs a machine with Python, torch and Ultralytics. The owner's machine may not run Python builds. Who runs it, where, and the file it writes: D12. Stages b and c cannot be checked without it. |
| b. YOLO26n forward on imported weights equals the reference | max abs difference on the 12 head outputs for one image; parameter count against the yaml comment (2,572,280, yaml:13; not recomputed) | `SILU` op and `ACTIVATIONS` row; rank-3 `MATMUL` (builder, shape rule); attention softmax as `exp(log_softmax(reshape))`; depthwise composition; composite blocks; `Detect` forward and decode; weight importer (section 3); host top-k postprocess | sushiai. No new SushiBLAS kernel (section 6.1) | ops, shape rules, blocks, importer, postprocess: now. The `SILU` lowering and the rank-3 `MATMUL` lowering: after migration, into the split tables. |
| c. One training step, loss equals the reference | six loss items against the reference on the same weights and batch; one conv-weight gradient; `running_mean` and `running_var` of one BN after the step | general loss root (`src/autograd/differentiator.cpp`, `src/autograd/rules/loss.cpp`); `STOP_GRADIENT`; `DIV`, `MIN`, `MAX`, `ATAN`, `ABS`, `SOFTPLUS` with rules; `SILU`, rank-3 `MATMUL` and softmax rules; BN variance exposure and statistics update (D13); host assigner with the small-box enlargement (tal.py:334-340); detection objective with the masked fixed-shape loss; multi-output step; per-step leaf feed; for D2 option 1: forward-only step, shared-parameter binding across two pools, objective-declared observed values | sushiai; sushiruntime if D2 picks `add_host` | differentiator seed, `STOP_GRADIENT`, ops, rules, host assigner (pure C++, testable against tal.py without an engine): now. Lowerings, BN node change, `Trainer` and `IObjective` changes, the second step: after migration. |
| d. Overfit a tiny dataset | loss falls and mAP50 approaches 1 on 8 to 16 images in fp32, with no mosaic or EMA | initialisers (ones, constant, per-parameter rule; `zero_final_layer` off); label parser, letterbox, flip; detection `Dataset` and its registry row; SGD or AdamW with parameter groups (`ParameterSlot`, `bind_parameters`); scheduler and warmup; host mAP evaluator; eval mode | sushiai | label parser, letterbox, flip, evaluator, scheduler logic: now. Initialisers (`init.cpp`), the `Dataset` class and row, groups, eval mode and the schedule hook in `Trainer`: after migration. |
| e. Full run | mAP50-95 on COCO val against the published YOLO26n figure (not read) | MuSGD; clipping; accumulation; EMA; mosaic, affine, HSV, `close_mosaic`; rectangular validation or its replacement (D7); checkpoint and resume with buffers, EMA and optimizer state | sushiai. No new SushiBLAS kernel (section 6.2) | augmentations and group rules: now. MuSGD update, clipping, accumulation, EMA, checkpoint changes (`io/checkpoint.hpp`): after migration. |

## 8. Decisions for the owner

### Decided on 2026-10-07

The owner ruled on these in conversation. The entries below still state the options; where they
recommend otherwise, this list wins.

- **Scope.** The target is both: YOLO26 trained from scratch, and its forward pass.
- **D4, depthwise convolution.** In scope. SushiAI asks SushiBLAS for the depthwise triple O1 of
  section 6.3. The softmax pair O2 is not decided.
- **D8, mixed precision.** On for the first run. The parity stages b and c stay in fp32, because a
  difference from the reference must be attributable; mixed precision is switched on from stage d.
  This adds work the entries below do not list: a precision decision for every operation YOLO26
  adds (convolution, the applied batch normalisation and SiLU narrow; batch statistics, the
  attention softmax, the whole loss and label assignment stay wide, as a proposal not yet checked
  against what the reference casts under AMP), HALF support in every kernel on the narrowed path
  (section 6.1 says no item needs HALF; under this decision F1, F2, F4 to F9 and O1 do), and the
  whole-step overflow skip that `docs/reference/KNOWN_ISSUES.md` records as per-parameter today.
- **D12, reference production.** An agent produces the reference files in a conda environment on
  this machine, by running the pinned Ultralytics commit and saving what it outputs, never by
  stating numbers.
- **D2, label assignment.** On the host, in two passes: a forward-only replay, the assignment in
  C++, then the full step.
- **D3, loss formulation.** Fixed-shape and masked, as this document recommends. The owner first
  chose the gather-based form and reversed it the same day after the two were set side by side:
  under a compiled step a gather needs a fixed capacity and a mask of its own, so it removes no
  mask and adds a capacity that can overflow, and it asks SushiBLAS for three kernel families the
  masked form does not need. Section 6.3 item O4 is not requested. The masked form has one hazard
  the gather avoids: an unassigned row can produce 0/0 or inf - inf, and a product with a zero
  mask does not remove a NaN. The guard is a selection before the arithmetic, not a blend: the
  operands of an unassigned row are replaced with a fixed valid box through a `SELECT` operation
  whose gradient passes only where the mask is set. `SELECT` is a new IR operation with a
  backward rule; the SushiBLAS logic family already has the kernel. The guard is reasoned on
  paper and has not run.
- **D7, validation.** Both paths: the one-to-one head with top-k, and the one-to-many head with
  NMS. NMS is host code and is new work. Both numbers are reported.
- **D6, augmentation.** The default pipeline in full: mosaic, random affine, HSV, horizontal flip,
  mosaic off for the last ten epochs. The pieces that are off by default are not written.
  Statistical equivalence with the reference, not bit parity with OpenCV.
- **D6, image decoding.** A separate tool under `apps/` decodes the dataset once into a raw cache
  with libjpeg-turbo, taken through vcpkg. The library reads the raw cache and takes no decoder
  dependency.

- **D13, batch normalisation statistics.** The batch variance becomes a fourth output of
  `BATCHNORM2D`, and the running statistics are updated beside the graph by their own recorder.
- **First tag.** SushiAI 0.1.0 is cut when this programme is finished.

### Open

- **D1. Layout.** Keep NHWC throughout and convert weights at import, or add axis-parameterised ops. *Recommend NHWC plus an importer.* Every spatial emitter rejects NCHW (bc:527-529), and attention needs no permute in NHWC (section 2).
- **D2. Where label assignment runs.**
  1. Host, with two passes: a forward-only step, then the full step. It costs a second forward per step and needs the five units of section 4: a second compiled step with its own pool, shared-parameter binding, a forward graph without the BN statistics update, objective-declared observed values and a host stage in `Trainer`. It does not work with today's types unchanged.
  2. Host, inside one plan through the runtime's `add_host`. One forward. It contradicts the rule at docs/design/ROADMAP.md:617-619 as worded ("no synchronous host read of tensor data anywhere inside forward/backward").
  3. Device IR ops. Adds item O3 of section 6 to SushiBLAS.

  *Recommend 1 for stages c and d, and 2 as the target once the roadmap sentence is ruled on.* Not 3. The eager front end does not help in its first version: it cannot differentiate across the read (EAGER_FRONT_END/SPEC.md:188-190).
- **D3. Loss formulation.** Fixed-shape masked sums per level (section 4), or gather-based as in the reference. *Recommend fixed-shape.* It removes dynamic shapes, gather, concat on anchors and broadcasting. It costs CIoU on every anchor per branch (8400 at 640 px) and the prediction blend that keeps off-foreground rows finite.
- **D4. New SushiBLAS kernels.** Neither is required. O1, a depthwise triple, replaces a composition that allocates 9 times the layer's activation twice. O2, a softmax pair, replaces a nine-task composition. This is the SushiBLAS owner's boundary (sushiblas: engine/README.md:227-230 keeps the word convolution out of the spatial domain). *Recommend asking for O1 and deferring O2 until a measurement shows the attention softmax matters.*
- **D5. Node seam.** One seam for every node of the model graph, `span<const ValueId>` in and several values out with a factory over `span<const Shape>`, plus an adapter for `ILayer`; or a change to `Module::forward` itself. *Recommend the separate seam.* Existing layers and tests stay untouched, and the container never tests a node's kind.
- **D6. Augmentation fidelity and the decode dependency.** Bit parity with OpenCV, or statistical equivalence with our own resize, warp and HSV. *Recommend statistical equivalence and a small image-decode dependency*, with mosaic, affine and HSV all present before stage e.
- **D7. Validation path.** Two choices.
  - *Head.* Default in-training validation in the reference uses the one-to-many head and external NMS: the validator sets `model.end2end = self.args.nms is False` (validator.py:167-168), `nms:` is empty (default.yaml:61), `_end2end` defaults to false (head.py:158) and inference then picks `one2many` (head.py:201). The top-k path runs only with `nms=False`. *Recommend the one-to-one head with top-k for SushiAI.* It is the deployed path and avoids writing NMS. The published figure stage e compares against must be read with this difference in mind.
  - *Batch shape.* Rectangular batches as in the reference (one compiled step per distinct shape), or a fixed 640x640 letterbox. *Recommend fixed square for stages d and e, stated as a departure.*
- **D8. Mixed precision.** *Recommend fp32 for the first run.* The reference supports it (default.yaml:36), and the eager spec defers loss scaling.
- **D9. The tripled lr on class branches.** It is commented "when finetuning" but applied whenever MuSGD is the optimizer, to every parameter of `cv3` and `one2one_cv3` (trainer.py:1217-1231). *Recommend copying the behaviour*, since that is what the reference does in a from-scratch run.
- **D10. SushiBLAS interface target.** *Recommend writing host-lane units now and every lowering once, against v0.1.0, after `lowering.cpp` is split into per-domain tables.* The alternative writes them twice and extends the if-chain.
- **D11. Pin the Ultralytics commit.** *Recommend `ce13994`* for every reference value in stages b and c.
- **D12. Reference production.** Stages b and c need files only Python with torch and Ultralytics can produce (stage r). To decide: who runs the exporter and on which machine, given that the owner's machine may not run Python builds; whether the exporter lives under `tools/`; and the file it writes. Two file options: SushiAI's `SACP` checkpoint written directly by the exporter, which needs no C++ reader but puts the layout permutations and name mapping in Python; or a neutral tensor file plus a C++ importer. A new file format is a public interface.
- **D13. BatchNorm statistics.** The batch variance as a fourth output of `BATCHNORM2D`, or the running update recorded inside the BN lowering with the buffers as node operands (section 2). *Recommend the fourth output*: the update stays a separate node, and the forward-only graph of D2 option 1 omits it by not emitting that node.

## 9. Unverified

- **Nothing has run.** No SushiAI conv path has executed (SPEC:59-60). Every "exists" is a source reading.
- **Derivations, on paper only.** The depthwise composition; the NHWC attention form; `exp(log_softmax(reshape))` as the attention softmax; the fixed-shape loss, the `K` leaf of the L1 term and the prediction blend against NaN; `relu` for `clamp_(0)`; Newton-Schulz per parameter through `gemm` transpose flags; `sum_rows(reshape(x, [n, 1]))` as the total sum. A unit test per item settles each.
- **Torch, not opened.**
  - Whether BatchNorm's running variance uses the unbiased n/(n-1) estimate. The stage c comparison of `running_var` settles it.
  - The default `nn.Conv2d` weight initialiser, and whether `kaiming_uniform` with `LEAKY_RELU` and slope √5 reproduces it (init.cpp:43, 60-62).
  - The gradient of `minimum` and `maximum` on ties, and of `abs` at zero. Section 6 item T7 gives one tie rule; parity needs torch's.
  - BCE, SiLU, Upsample and MaxPool definitions.
- **Ultralytics, not read.**
  - `nms.py`.
  - The `eps` default of `bbox_iou` (metrics.py:108).
  - The `pad` value in the rectangular batch shape (U/data/build.py).
  - `data/utils.py`, the augmentation code and the mAP code beyond the lines cited.
  - `parse_model`, and the `C2PSA` constructor that fixes the attention `dim` at the n scale.
  - Whether any identifier named `ProgLoss` or `STAL` exists in the tree. The mechanisms this map relies on are `E2ELoss.decay` (loss.py:1352-1354) and `stride_val` (tal.py:59, 334-340).
- **Parity and reference numbers.**
  - fp32 Newton-Schulz against the reference's bfloat16 (muon.py:39): no source establishes equivalence.
  - The n-scale parameter count was not recomputed.
  - The published YOLO26n mAP, its training configuration and its validation head were not read.
- **SushiAI and SushiBLAS internals.**
  - Whether the elementwise kernels and `blas().copy` accept rank-4 operands. Kernel bodies under `sushiblas/src` were not read.
  - Whether `im2col` honours a strided input. SushiAI passes contiguous inputs, so the map does not depend on it.
  - Whether two `CompiledStep`s can share parameter storage through `TensorPool::bind` (lowering.hpp:101), which D2 option 1 needs. Reading `CompiledStep` and the arena ownership settles it.
- **Runtime.** `add_emitter` and `add_host` behaviour on the CUDA build. A host access to a shared buffer while a kernel runs on the same GPU faults on Windows with CUDA (`D:\Projects\sushiblas\docs\agent\2026_10_07_V010\SUSHIAI_CONSTRAINTS.md:47-51`), which bears on D2 options 1 and 2: the host stage must read only after the plan has drained.
- **Stage a.** YOLOv3-tiny's own layer list was not read. Its stride-1 2x2 pool with asymmetric padding, if present, is unchecked against `MaxPool2d`'s symmetric-pad emitter (bc:579-582).
