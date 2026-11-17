# tests/vision/datasets/test_darknet_pipeline.cpp


The pipeline of Darknet's detector augmentation: its name, the identity settings, a sample against
the draws keyed by its seed, epoch and index, its independence of the batch, a new epoch, a mask
and the refusals.

Suites: `Small_DatasetsDarknetPipeline`.

Tests `DarknetPipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/darknet_pipeline.hpp`. The cases read
three raw RGB images from memory, of the sizes 20 by 24, 24 by 20 and 32 by 32, whose bytes rise
by 3 from 0, 50 and 100; sample i holds one object of class i that spans [0.25, 0.75] of each
side. The input is 16 by 16 and the seed 20261115. The settings that augment are jitter 0.3,
hue 0.1, saturation 1.5 and exposure 1.5.

## ItIsNamedDarknet

`name` is "darknet".

## TheIdentitySettingsOnlyResize

With jitter 0, hue 0, saturation 1 and exposure 1 each sample equals its source resized to 16 by
16 by `resize_darknet`, byte for byte, and its one object keeps its class and spans [4, 12] on
both axes.

## ASampleIsTheAugmentationOfItsKeyedDraws

Sample 1 in epoch 3 equals `apply_darknet_augment` of the plan `plan_darknet_augment` draws from
`make_sample_random(20261115, 3, 1)`, and holds file 1.

## ASampleDoesNotDependOnItsBatch

In epoch 2 sample 1 is the same image whether produced in the batch 0, 1, 2, in the batch 2, 1, 0
or alone, and sample 0 is the same in the first two.

## ANewEpochChangesTheImage

Sample 2 differs between epoch 0 and epoch 1.

## ASampleWithAMaskIsRefused

A list whose sample 0 has a mask fails `produce` with `InvalidDescription`, "a.saim", "has a
mask" and "'darknet'".

## MakeRefusesWhatItCannotBuild

`make` fails with `InvalidDescription` for an input of 0 by 8 ("DarknetPipeline", "0 by 8"), for
5 channels ("5 channels") and for a jitter of 0.5 ("DarknetPipeline", "the jitter is 0.5").

## ProduceRefusesAnEpochAndCountsItCannotServe

`produce` fails with `InvalidDescription` and "epoch -1" for the epoch -1, and with "2 samples" and
"1 fitted" for two indices and one slot.
