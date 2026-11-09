# tests/vision/training/test_free_loss_gains.cpp


Suites: `Small_FreeLossGains`.

Tests `read_free_loss_gains` of `free_loss_gains.hpp`, which reads the four gains of the anchor-free
loss from an objective section. The section is built in memory; no case needs a device or a file.

### AnAbsentKeyKeepsItsDefault

A section of the type alone gives a box gain of 7.5, a class gain of 0.5, a distribution gain of
1.5 and an L1 gain of 0.

### EveryStatedGainIsRead

A box gain of 2, a class gain of the whole number 1, a distribution gain of 0.25 and an L1 gain of
1.5 are read as stated.

### ANegativeGainIsRefusedNamingItsKey

A class gain of -0.5 fails with `InvalidDescription` naming `class_gain` and "not negative".

### AGainThatIsNoNumberIsRefusedNamingItsKey

A box gain of the string "high" fails with `InvalidDescription` naming `box_gain`.

### AnInfiniteGainIsRefusedNamingItsKey

An L1 gain of infinity fails with `InvalidDescription` naming `l1_gain` and "finite".
