# tests/vision/importers/test_caffe_import.cpp

Suites: `Small_CaffeImport`, `Small_CaffeEltwiseOperation`, `Medium_CaffeLenetImport`.

The `Small_` cases read a prototxt written in the test and give its layers blobs of floats the
test chooses; no file is opened. The `Medium_` cases read `tests/vision/importers/fixtures/caffe/`,
which `tools/reference/make_caffe_lenet_fixture.py` and `tools/reference/export_caffe_reference.py`
write, and skip when a file is missing. Every expected value is derived in the case that holds it.

## BatchNormAndScaleBecomeOneBatchnorm2dRow

A BatchNorm layer and the Scale layer after it write one `batchnorm2d` row with the epsilon of
`batch_norm_param`, and the model publishes the Scale's top.

## BatchNormAndScaleStoreTheFoldedValuesExactly

Mean {2, 4} and variance {4, 8} under the scale factor 2 are stored as {1, 2} and {2, 4}; the
Scale's scale {3, 5} and shift {0.5, -1} are stored as `weight` and `bias`. All values are dyadic,
so the comparison is exact.

## ABatchNormWithoutAScaleStoresUnitScaleAndZeroShift

A BatchNorm layer that no Scale layer reads stores `weight` 1 and `bias` 0 per channel.

## AZeroScaleFactorStoresZeroStatistics

A scale factor of 0 stores a mean and a variance of 0, as Caffe's own inference does.

## AScaleThatNoBatchNormPrecedesIsRefused

A Scale layer that reads no BatchNorm top is refused with `Unsupported`, naming the layer.

## UnsupportedLayersAreListedTogether

A Deconvolution and an LRN layer in one net give one `Unsupported` message that names both and
counts two.

## ALayerOfTheTrainingNetAloneIsLeftOut

A Data layer with `include { phase: TRAIN }` is neither translated nor listed as unsupported.

## ALayerThatReadsAnUnwrittenBlobIsRefused

A layer whose bottom no layer above it writes fails with `InvalidDescription`, naming the layer and
the blob.

## ANetWithoutLayersIsRefusedForItsMissingInput

A prototxt with a name and nothing else is refused for declaring no input.

## ConvolutionBecomesAConv2dRowOfTheStatedWindow

`num_output: 3 kernel_size: 3 stride: 2 pad: 1 bias_term: false` becomes a `conv2d` row with those
filters, kernel, stride and padding, `use_bias` false, over the model input of 4 by 4 by 2.

## ConvolutionWeightsMoveToTheRecordLayout

One filter over 2 channels and a 1 by 2 kernel, stored by Caffe as 0 1 | 2 3, is stored as
0 2 1 3 with the shape [1, 1, 2, 2]: `[out, kh, kw, in]`.

## ConvolutionGroupsBecomeTheGroupsKey

`group: 2` over 2 channels and 4 filters writes a `conv2d` row whose key `groups` is 2.

## ConvolutionOfOneGroupStatesNoGroupsKey

A convolution with no `group` writes no `groups` key.

## GroupedConvolutionWeightsKeepOneChannelPerFilter

Four filters over two channels in two groups read one channel each, so the blob 1, 2, 3, 4 is
stored as the weight [4, 1, 1, 1] with the four values in order.

## DepthwiseConvolutionWeightsAreKernelHeightWidthChannel

Two filters over two channels in two groups, with a 1 by 2 kernel, form a depthwise convolution.
Filter 0 holds the taps 10, 11 and filter 1 the taps 20, 21, and the stored weight [1, 2, 2] lists
tap 0 as (10, 20) and tap 1 as (11, 21).

## ConvolutionGroupsThatDoNotDivideTheChannelsAreRefused

`group: 3` over 2 channels fails with `InvalidDescription`, naming the group and the 2 channels.

## ConvolutionOfZeroGroupsIsRefused

`group: 0` fails with `InvalidDescription` naming "group of 1 or more".

## AMaxPoolOverAnOddExtentGetsTrailingPadding

A 2 by 2 window with stride 2 over 5 by 5 yields 3 windows in Caffe, which rounds up. The
`maxpool2d` row states `trailing_padding` [1, 1] so its floor-based count gives 3 as well.

## AnAveragePoolPastThePaddedEdgeIsRefused

The same window as an average pool is refused: Caffe divides the last window by a smaller area than
`avgpool2d` does.

## AnAveragePoolThatFitsBecomesAnAvgpool2dRow

An average pool whose windows all lie inside the input becomes an `avgpool2d` row with no trailing
pad.

## GlobalAveragePoolingBecomesAGlobalAvgpoolRow

`global_pooling: true` with `pool: AVE` becomes a `global_avgpool` row.

## ReluBecomesAReluRow

A ReLU with no slope becomes a `relu` row.

## AReluWithASlopeBecomesALeakyReluRow

`negative_slope: 0.125` becomes a `leaky_relu` row with that slope.

## AnInPlaceReluPublishesTheBlobFromItsOwnRow

A ReLU whose top is its bottom rewrites the blob: the model publishes `conv` from the ReLU's row
(row 1), not from the convolution's.

## AnInnerProductAfterAnImageFlattensFirst

An InnerProduct over a 4 by 4 by 2 blob writes a `flatten` row, then a `linear` row of the stated
units.

## InnerProductWeightsFollowTheChannelsLastFlatten

Over 2 channels of 2 by 2, the Caffe weight 0..7 is stored as 0 4 1 5 2 6 3 7, the order of a
flattened [H, W, C] image.

## ConcatJoinsTheChannelsOfItsBottoms

Two ReLU rows joined along the default axis 1 become one `concat` row that reads rows 0 and 1.

## ConcatAlongAnotherAxisIsRefused

`axis: 2` is refused with `Unsupported`; the `concat` row joins the channels.

## EltwiseSumOfThreeBottomsChainsTwoAddRows

The default SUM over three bottoms writes two `add` rows; the second reads the first and the third
bottom.

## AnEltwiseCoefficientOtherThanOneIsRefused

`coeff: 1 coeff: -1` is refused with `Unsupported`; the `add` row sums with weight 1.

## TheOperationBecomesAChainOfItsRow

Suite `Small_CaffeEltwiseOperation`, one row per operation. PROD over three bottoms writes two
`mul` rows and MAX writes two `maximum` rows; the second reads the first and the third bottom.

## ACoefficientOnTheOperationIsRefused

Suite `Small_CaffeEltwiseOperation`. `coeff` on a PROD or a MAX fails with `InvalidDescription`
naming the layer, the coefficients, the operation and "SUM only", as Caffe refuses it.

## AnUnknownEltwiseOperationIsRefused

`operation: MEAN` fails with `InvalidDescription` naming MEAN and "SUM, PROD and MAX".

## ASoftmaxOverAVectorBecomesASoftmaxRowAndIsPublished

A Softmax after an InnerProduct writes a `softmax` row with no `axis` key. The model publishes the
blob `prob` from row 2, and no note is written.

## ASoftmaxOverTheChannelsOfAnImageStatesNoAxis

A Softmax over the default axis 1 of a [N, 2, 4, 4] blob writes a `softmax` row with no `axis` key,
since the channels are the last axis of a sample.

## ASoftmaxOverTheRowsOfAnImageStatesTheRowAxis

`axis: 2` of [N, C, H, W] is the rows, the first axis of a sample [H, W, C], so the row states
`axis` 0.

## ASoftmaxThatAnotherLayerReadsIsAnInteriorRow

A Softmax whose top feeds an InnerProduct is a `softmax` row between two `linear` rows, and only
the last blob is published.

## ASoftmaxOverTheBatchAxisIsRefused

`axis: 0` fails with `Unsupported` naming the batch axis.

## ASoftmaxOverAnAxisBeyondTheBlobIsRefused

`axis: 4` over a rank-4 blob fails with `InvalidDescription` naming axis 4.

## DropoutKeepsItsRatioAsTheRate

`dropout_ratio: 0.75` becomes a `dropout` row of `rate` 0.75; the row is the identity at inference.

## AWeightlessNetImportsWithoutTensors

A net whose layers hold no blobs imports its rows and lists no tensor.

## ABlobThatDoesNotFillItsLayerIsIncompatible

Three floats for a weight that the description gives two is `Incompatible`, naming the layer and
both counts.

## ALayerWithoutBlobsInAWeightedNetIsIncompatible

When one layer holds blobs, a convolution without any is `Incompatible`, naming the layer.

## ThreeFilesAreRefusedBeforeAnyIsOpened

`import_caffe_files` takes a prototxt and optionally a caffemodel; three paths fail with
`InvalidDescription` before a file is read.

## TheImportedLenetHoldsTheRowsAndTheTensorsOfItsFiles

Suite `Medium_CaffeLenetImport`. The fixture's LeNet imports to the nine rows its prototxt lists,
ending in `softmax`, eight tensors (a weight and a bias for each of four layers) and the output
`prob`.

## TheProbabilitiesAreTheSoftmaxOfOpenCvsScores

The imported LeNet, run on the fixed input of `lenet.expected.json`, gives the blob `prob`. Each
probability is the exponential of the score `cv2.dnn.readNetFromCaffe` computed, over the sum of
the exponentials of the ten scores. The bound is 2 * 285 unit roundoffs times the largest score,
plus 16 for the softmax: a score moves by 285 unit roundoffs times the largest score, since the
longest path sums 25 + 100 + 128 + 32 products, and a probability moves by at most twice that.
