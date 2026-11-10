# tests/vision/training/test_free_bias_init.cpp


Suites: `Small_TrainingFreeClassBias`, `Small_TrainingFreeBiasInit`, `Medium_TrainingFreeBiasInit` and
`Medium_TrainingFreeBiasInitCopies`.

Tests `free_class_bias` and `initialize_free_biases` of `free_bias_init.hpp`, which start the last
box and class biases of an anchor-free detect head. The traced-head cases allocate on the CPU SYCL
device through `EngineTest`, so they are medium. The refusal cases build their state from entries
on the engine of a `HostExecutionBackend` and are small. The traced head has three levels of 8, 4 and 2
cells over a 64 by 64 input, three classes and both copies, under the state prefix `23`.

### MatchesTheClosedForm

The class bias of stride 8 over 80 classes equals `log(5 / 80 / 6400)`, to 16384 double unit
roundoffs. The closed form is the only place the class bias is checked: the bias of each level is
the same function of its stride.

### TheLastBoxBiasIsOne

Rows: levels 0, 1 and 2 of the one-to-many copy and of the one-to-one copy. After
`initialize_free_biases` every element of the last box bias of the row, `23.[one2one_]cv2.<i>.2.bias`,
is 1, to 32 float32 unit roundoffs.

### RefusesAPrefixTheStateLacks

A prefix `22` that the state does not hold fails with `InvalidDescription` naming
`'22.cv2.0.2.bias'`, and the entry `23.cv2.0.2.bias` is still 0.

### AnEntryOfAnotherExtentIsRefusedAndNoBiasChanges

A state of one level whose class bias has 4 elements against the 3 classes fails with
`InvalidDescription` naming `'p.cv3.0.2.bias'`, `[4]` and `give [3]`. The box bias, which was
planned first, is still 5.

### AnEntryOfAnotherElementTypeIsRefusedAndNoBiasChanges

A class bias held as float64 fails with `InvalidDescription` naming `'p.cv3.0.2.bias'` and "is not
float32". The box bias is still 5.
