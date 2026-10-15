# modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_mask.hpp


`load_sample_mask` decodes the file a sample's `mask` names through `Codecs::load_image` with
`ChannelLayout::Stored` and `PaletteRead::Indices`, so a palette PNG gives its indices and a
gray PNG its bytes. It is `InvalidDescription` for a sample index outside the list, a sample
with no mask and a mask index outside `files`; any other failure is the decoder's.
