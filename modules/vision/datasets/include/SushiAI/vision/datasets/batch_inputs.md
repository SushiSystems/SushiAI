# modules/vision/datasets/include/SushiAI/vision/datasets/batch_inputs.hpp


`write_batch_inputs` writes image `n` of a batch into row `n` of a float32 `[N, H, W, C]` tensor,
each byte converted by a `Config::InputNormalise`: `(level * scale - mean) / std` per channel, after
the channels are put in the stated order. The default normalisation gives the byte divided by 255
bit for bit, the scaling of `source/mnist.cpp`. Every collator holds the normalisation it was built
with and passes it here; a data section's `normalise` key reaches it through `CollatorRequest`. A
channel order, a mean or a deviation needs a 3-channel input, and the call is `Incompatible` for
another count. It is `Incompatible` when the tensor
is not float32 of rank 4, when `N` is not the sample count, or when an image is not `H` by `W` of
`C` channels; the message gives the image's position in the batch.
