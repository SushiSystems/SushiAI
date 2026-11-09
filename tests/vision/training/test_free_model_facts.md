# tests/vision/training/test_free_model_facts.cpp


Suites: `Small_TrainingFreeModelFacts` and `Small_TrainingFreeModelFactsLevel`.

Tests `derive_free_model_facts` of `free_model_facts.hpp`, which reads the levels, the strides, the
`reg_max` and the class count of an anchor-free model from the ports it publishes. The fixture is a
`DetectHead` over three levels of 8, 4 and 2 cells and 16, 32 and 64 channels, three classes and a
batch of 2, over a 64 by 64 input. The ports are published as the head names them. No case needs a
device.

### AThreeLevelHeadGivesItsCountsAndTheOneToOneFlag

A head with both copies gives `reg_max` 1, three classes, three levels and a one-to-one copy.

### ALevelGivesItsStrideAndSides

Rows: levels 0, 1 and 2. The level has the stride 8, 16 or 32, which is 64 over its side, and the
sides 8, 4 or 2.

### ALevelNamesThePortsOfBothCopies

Rows: levels 0, 1 and 2. The level names `one2many.level<i>.boxes` and `.scores` and
`one2one.level<i>.boxes` and `.scores`.

### ASingleCopyHeadGivesItsRegMax

A head with `reg_max` 16 and no end-to-end copy gives `reg_max` 16, no one-to-one copy, three
levels, and empty one-to-one port names.

### RefusesALevelThatDoesNotDivideTheInput

An input of 60 by 60 fails with `InvalidDescription` naming `'one2many.level0.boxes'`, "8 by 8" and
"60 by 60".

### RefusesAnInputThatIsNotSquare

An input of 64 by 32 fails with `InvalidDescription` naming `'one2many.level0.boxes'` and "equal
along H and W".

### RefusesOutputsWithoutTheDetectPorts

Outputs named `boxes` alone fail with `InvalidDescription` naming `'one2many.level0.boxes'`.
