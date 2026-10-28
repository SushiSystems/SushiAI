# tests/vision/training/test_free_detection_loss_host.cpp

Suites: `Small_FreeDetectionLossHost`.

Tests `evaluate_free_detection_loss` of `free_detection_loss_host.hpp` against closed forms; `test_free_detection_loss.cpp` uses the same function as the oracle of the graph. A row has an anchor point at (1, 1), distances of 1 that decode to the box (0, 0, 2, 2), the same target and safe box, a weight of 0.5 and one class. Terms are held to 1e-12, and the box term to 1e-6 because CIoU adds epsilons of 1e-7.

### TheClassTermIsTheCrossEntropyOverTheScoreSum

For two classes with the logits 0 and 2, the targets 0 and 1 and a score sum of 2, the class term is 0.41003759580145904.

### ABoxEqualToItsTargetCostsNothing, TheBoxTermIsOneMinusCIoUTimesTheWeight, AMaskedRowTakesTheSafeBoxInPlaceOfItsOutput

A decoded box equal to its target gives a box term of 0. The box (0, 0, 1, 1) in (0, 0, 2, 2) has an IoU of 1/4, a centre term of 1/16 and equal aspects, so its term is (1 - 3/16) x 0.5 = 0.40625. A row with a mask of 0 takes its safe box and gives 0 whatever its output holds.

### TheDistributionTermIsTheCrossEntropyAgainstTheBinWeights, OneBinPerSideHasNoDistributionTerm

At two bins per side, equal logits and the weights 1/4 and 3/4, the term is 4 log 2 = 2.772588722239781. At one bin per side it is 0.

### TheL1TermWeighsEachSideErrorByItsScale, ANoL1GainNeedsNoL1Arrays

With an L1 gain of 1, distances (1, 1, 0, 0), a target of 0 and a scale of 0.5, the term is 1. With no L1 gain the term is 0 and the L1 arrays may be empty.

### TheTotalIsTheGainedTermsTimesTheBatch, LevelsAddTheirTerms

With the gains 2, 3, 4 and 5 and a batch of 3 the total is 3 times the gained sum of the terms. Two equal levels double every term and the total.

### AnArrayOfAnotherSizeIsRefusedNamingIt, ASettingThatIsNotPositiveIsRefused, NoLevelGivesNoLoss

A scores array of 3 values for 2 classes and a box mask of 2 values are refused naming "scores" and "box mask". A `reg_max`, a class count or a batch of 0 and a score sum of 0 are refused. No level gives 0 in every term.
