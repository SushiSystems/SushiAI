# tests/layers/interchange/test_caffe_net.cpp

Suites: `Small_CaffeNet`.

Tests the readers and the join of `modules/layers/interchange/include/SushiAI/interchange/caffe_net.hpp`.
The prototxt is text in the test; the caffemodel is assembled byte by byte with the helpers of
`protobuf_bytes.hpp` using the field numbers of `caffe.proto` (NetParameter 1, 2, 100;
LayerParameter 1, 2, 3, 4, 7; V1LayerParameter 2 to 6; BlobProto 1 to 5, 7, 8; BlobShape 1). The
net has a Convolution `conv1` (weights [2, 1, 3, 3], bias [2]), a V1 `INNER_PRODUCT` layer `ip1`
(legacy extents 1, 1, 3, 2) and a weights-less `relu1` in the model.

## ReadsTheLayersOfAPrototxt

Name, the net-level input with its four `input_dim` numbers as the shape [1, 1, 4, 4], and the
layer `conv1` with type, bottoms, tops and its `convolution_param` message (`num_output` is 2).

## ReadsAV1LayerAsALayer

`layers { type: INNER_PRODUCT }` reads as the layer `ip1` of type `InnerProduct`, marked legacy.

## ReadsInputShapesFromInputShapeMessages

Two `input_shape` messages give [1, 3] and [2].

## ReadsTheBlobsOfACaffemodel

`conv1` holds two blobs; the weights extents are [2, 1, 3, 3], 18 floats whose bytes are a view at
the offset where the packed payload first appears in the file, with element 17 equal to 8.5; the
bias reads 20.0 at index 1. The V1 layer reads as legacy, type `InnerProduct`, with the extents
[1, 1, 3, 2] from the four legacy fields.

## ReadsFloatsThatAreNotPacked

Three fixed32 fields read into an owned blob of three floats.

## JoinsWeightsToLayersByName

After the join `conv1` holds its two blobs, `ip1` its one, the weights are the same bytes the
caffemodel read, and the layer's prototxt message is still there.

## NamesTheWeightsLayersAbsentFromTheNet

A caffemodel with two more layers carrying blobs, `fc9` and `fc10`, fails with `Incompatible`
naming both, and the net has no blobs afterwards.

## IgnoresAWeightslessLayerTheNetLacks

A loss layer with no blobs and no place in the net joins without error.

## RefusesWeightsThatDoNotFitTheLayer

A weights layer `conv1` of type `InnerProduct` fails naming the layer and both types; a net with
two layers named `conv1` fails naming the count.

## RefusesABlobWhoseFloatsDoNotFillItsExtents

Three floats for the extents [2, 2] fail with `InvalidFile` naming the layer, 3 floats and 4.

## RefusesBlobsItCannotReadAsFloats

A blob in `double_data` fails with `Unsupported`; a packed field of six bytes fails with
`InvalidFile`.

## RefusesAV1TypeNumberTheTableLacks

A V1 layer of type 99 fails naming the layer and the number.

## ReadsTheV1TypeNumberOfADeconvolution

A V1 layer of type number 39, which `caffe.proto` names `DECONVOLUTION`, reads as the type
`Deconvolution`.

## RefusesAPrototxtThatStatesALayerBadly

A layer with no type, an unknown V1 enumerator, an `input_dim` count that is not four per input, a
dimension that is no integer and an unclosed brace each fail with `InvalidFile` naming the file
and, where the text gives one, the line.
