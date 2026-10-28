# modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp


## expand_gray

One channel in, three equal channels out. A source with another channel count is `InvalidDescription`; a source that fails `check_source` stops the process. The
pipelines resize a grey image first and expand it after, which gives the same bytes as the
other order because the resize treats each channel alone.
