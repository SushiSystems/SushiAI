# modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp


## expand_gray

One channel in, three equal channels out. A source with another channel count throws. The
pipelines resize a grey image first and expand it after, which gives the same bytes as the
other order because the resize treats each channel alone.
