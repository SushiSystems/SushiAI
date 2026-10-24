# tests/vision/datasets/test_sample_cache.cpp


Suites: `Small_DatasetsSampleCache`.

Tests `write_sample_cache`, `use_sample_cache` and `sample_cache_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_cache.hpp`. The source images
are raw image files the cases write under the system's temporary directory, and the cache is
the folder `cache` beside them.

## TheFirstPassWritesAndTheSecondReuses

Over two RGB images of 2 by 3 and 4 by 2 pixels, the first `write_sample_cache` reports 2
written, 0 reused and the byte count of the two raw image files. The second reports 0 written,
2 reused and 0 bytes.

## AListPointedAtTheCacheLoadsTheSamePixels

After a write, `use_sample_cache` gives a list whose one file lies in the cache directory, has
the extension `.saim` and a name that starts with "a-". `load_sample_image` over it gives the
size, the three channels and every byte of the source image.

## ALongestSideReducesOnlyWhatExceedsIt

With `longest_side` 4, an image of 4 rows and 8 columns at level 200 loads from the cache as 2
rows and 4 columns with a first byte of 200. A 2 by 2 image loads as 2 by 2.

## AnImageThatChangedIsStale

After the source is rewritten with another size, `use_sample_cache` fails with `FileNotFound`,
"images/a.jpg", "sample cache" and "changed since". A second write reports 1 written and 0
reused, and the image then loads from the cache with a width of 3 and a first byte of 2.

## StoredPixelsAndMasksAreLeftAlone

For a list of one sample with stored pixels and one sample with an image and a mask, the write
reports 1 written. In the list `use_sample_cache` returns, the file of stored pixels and the
mask keep their paths and the image's path differs.

## ANameFollowsTheSourceAndTheOptions

`sample_cache_path` returns the same path twice for one source and one set of options, and its
file name is 23 characters long. With `longest_side` 320 the path differs. A source that does
not exist fails with `FileNotFound` and "gone.jpg".

## TheNameIsTheFNV1aHashOfTheKeyText

The digits of the file name equal the 64-bit FNV-1a hash of `<path>|<bytes>|<time>|<side>|<layout>`.
The case spells the hash out beside the module and checks that spelling against a value computed
in Python for "images/a.jpg|1234|5678|320|0". The time depends on the platform clock, so the
value itself cannot be fixed in the case. The name must not change, because a cache written under
the old name would be written again.

## RefusesOptionsItCannotServe

`write_sample_cache` fails with `InvalidDescription` for options without a directory
("no directory"), for `longest_side` -1 ("a longest side of -1") and for `longest_side` 1 over
a list whose sample has a mask ("a longest side of 1", "masks").
