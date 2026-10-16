# tests/layers/nn/test_loss.cpp


Suites: `Small_CrossEntropyLoss`.

Tests the nodes `CrossEntropyLoss` emits and the operands it refuses.

This file needs no device. The gradient a cross-entropy root produces is pinned in
`test_layers.cpp`; the cases here pin the forward nodes alone.

## TheLossIsTheLikelihoodOfALogSoftmax

Over [4, 3] logits and a [4, 3] target the loss appends exactly two nodes. The first is a
K_LOG_SOFTMAX that reads the logits; the second is a K_NLL_LOSS that reads the log-softmax result
and then the target, and writes the returned loss.

## TheLossIsOneElementOfTheLogitsType

With float64 operands the loss holds one element and is float64 as well.

## OperandsOfTwoShapesAreRefused

A [4, 2] target beside [4, 3] logits throws an error naming NLL_LOSS and both shapes.
