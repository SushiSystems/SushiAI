# tests/vision/datasets/test_sample_cache.cpp


Suites: `Small_DatasetsSampleCache`, `Medium_DatasetsSampleCache` and
`Small_DatasetsSampleCacheRefusal`.

Tests `write_sample_cache`, `use_sample_cache` and `sample_cache_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_cache.hpp`. The source images
of the `Medium_` cases are raw image files they write under the system's temporary directory,
and the cache is the folder `cache` beside them. Those cases are `Medium_`: they write files,
and the cache name holds the modification time the platform clock gives. The `Small_` cases
pass a `MemorySource` as the last argument, which holds `images/a.jpg` as the four bytes
"abcd" and gives every file the stamp 0, so the name of a cache file is the same on every
platform and no file is touched. The refusals decide on the options before any file is
examined, so they are `Small_` and name files that never exist.

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

`Small_`. `sample_cache_path` over the memory source returns the same path twice for one source
and one set of options, in the cache directory. Its file name is 23 characters long. With
`longest_side` 320 the path differs. A source the memory source lacks fails with
`FileNotFound` and "gone.jpg".

## TheNameIsTheFNV1aHashOfTheKeyText

`Small_`. The digits of the file name equal the 64-bit FNV-1a hash of
`<path>|<bytes>|<time>|<side>|<layout>`. Over the memory source the key of `images/a.jpg` at
`longest_side` 320 is "images/a.jpg|4|0|320|0", whose hash, computed in Python from the
offset basis 14695981039346656037 and the prime 1099511628211, is ec970685015c5c41, so the
file name is "a-ec970685015c5c41.saim". The name must not change, because a cache written
under the old name would be written again.

## AListIsPointedAtTheCacheFileTheSourceHolds

`Small_`. With the file `sample_cache_path` names added to the memory source,
`use_sample_cache` returns the list with that path in place of `images/a.jpg`.

## AnImageWithoutACacheFileIsRefused

`Small_`. Without a cache file in the memory source, `use_sample_cache` fails with
`FileNotFound`, "images/a.jpg", "sample cache" and "changed since".

## ACacheFileTheSourceHoldsIsReused

`Small_`. With the cache file in the memory source, `write_sample_cache` reports 0 written, 1
reused and 0 bytes, and creates no directory. The source is passed in a `DatasetStorage` with no sink.

## RefusesOptionsItCannotServe

`Small_DatasetsSampleCacheRefusal`, one row each: `write_sample_cache` fails with
`InvalidDescription` for options without a directory ("no directory"), for `longest_side` -1
("a longest side of -1") and for `longest_side` 1 over a list whose sample has a mask
("a longest side of 1", "masks").
