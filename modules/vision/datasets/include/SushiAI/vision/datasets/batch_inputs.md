# modules/vision/datasets/include/SushiAI/vision/datasets/batch_inputs.hpp


`write_batch_inputs` writes image `n` of a batch into row `n` of a float32 `[N, H, W, C]` tensor,
each byte divided by 255, the scaling of `source/mnist.cpp`. It is `Incompatible` when the tensor
is not float32 of rank 4, when `N` is not the sample count, or when an image is not `H` by `W` of
`C` channels; the message gives the image's position in the batch.
