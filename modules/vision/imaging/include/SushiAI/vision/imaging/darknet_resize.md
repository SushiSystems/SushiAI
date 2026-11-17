# modules/vision/imaging/include/SushiAI/vision/imaging/darknet_resize.hpp


## resize_darknet

Darknet's `resize_image` (`image.c` of the Darknet repository) resizes in two passes, the columns of every source
row first, then the rows of that result. Each pass reads the source at
`index * (in - 1) / (out - 1)` and blends the two neighbours linearly, so the corners of the
result are the corners of the source. `resize_bilinear` maps pixel centres,
`(x + 0.5) * in / out - 0.5`, and the two differ by up to one source pixel at every edge of a
reduction; a letterboxed image fed to a Darknet model needs this one.

Differences from the C code, each chosen so the result does not depend on how a float product
rounds:

- The positions and the blend are in double. Darknet's float `(h - 1) * h_scale` can fall one
  ulp short of the last source row, and its `continue` on the last row then drops the second
  neighbour. Here the last destination index is the last source index exactly.
- A destination side of 1 reads source position 0. Darknet divides by zero there.
- The result is a level in [0, 255], rounded half up; Darknet keeps a float in [0, 1].
