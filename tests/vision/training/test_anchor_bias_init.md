# tests/vision/training/test_anchor_bias_init.cpp


Suites: `Small_AnchorBiasInit`, `Medium_AnchorBiasInit`.

The formulas are pinned to values computed offline at 1e-12: the objectness bias at strides 8,
16 and 32, the class bias at 80 and 2 classes. The derivation reads a small model file whose
`head_0` is a `sigmoid` over row 0 and whose `head_1` is row 2 itself, over a 64 pixel input,
and gives the entries `0.bias` and `2.bias` at strides 4 and 8. A convolution without a bias, a
head that ends in a `maxpool2d` row, an output the model does not publish and a level of 7 cells
over 64 pixels are each refused with the output or the row named.

On an engine, two entries of 14 values started at 1.0 gain the objectness bias at entry 4 and
the class bias at entries 5 and 6 of both slots, and keep entries 0 to 3. One case reads each
head against the whole expected entry (`TheFirstHeadGainsTheBiasesOfItsStride`, stride 4, and
`TheSecondHeadGainsTheBiasesOfItsStride`, stride 8), within the float32 bound of
`K_BIAS_TOLERANCE`. An entry of 13 values refuses the call naming `2.bias` and `[14]`
(`AnEntryOfTheWrongExtentIsRefusedAndNamed`), and the refusal leaves the other entry at 1.0
(`AnEntryOfTheWrongExtentLeavesEveryBiasUnchanged`). An absent entry and a head count that
differs from the task's are refused.

`AnAbsentEntryIsRefusedAndNamed` and `AHeadCountThatDiffersFromTheTaskIsRefused` stay `Medium_`. Both
refusals are decided on names and counts, but `NN::ModelState::from_entries` takes an engine and
no host view of the entries exists; the seam is on the list of RW-155.
