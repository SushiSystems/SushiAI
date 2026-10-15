# modules/vision/datasets/include/SushiAI/vision/datasets/mask_png.hpp


`read_mask_png` lists the files with an image extension at any depth below
`root/<image_folder>/<split>`, sorted by path, and pairs each with the file of the same relative
path and the extension `.png` below `root/<mask_folder>/<split>`. The folders default to `images`
and `masks`; an empty `split` reads the two folders themselves. `files` holds the images and then
the masks, and a sample's `mask` is the index of its mask. No file is opened, so `class_count`
comes from the caller and no mask is checked against it.

`image_suffix` and `mask_suffix` pair names that differ. The mask of
`<image_folder>/<split>/a/b<image_suffix>.jpg` is `<mask_folder>/<split>/a/b<mask_suffix>.png`,
and both are empty by default. With a `mask_suffix`, only the `.png` files whose stem ends in it
are masks; the others in the mask folder are not listed. Cityscapes pairs with
`"images": "leftImg8bit"`, `"masks": "gtFine"`, `"image_suffix": "_leftImg8bit"` and
`"mask_suffix": "_gtFine_labelIds"`.

| Code | When |
| --- | --- |
| `InvalidDescription` | `class_count` is below 1 |
| `FileNotFound` | One of the two folders is not a directory; an image has no mask, with both paths |
| `InvalidFile` | The image folder holds no image; an image whose stem does not end in `image_suffix`, with its path; a mask has no image, with its path; two images share a stem and so a mask; a directory cannot be listed |
| `OutOfMemory` | An allocation fails |

A mask extension is compared as `.png` in lower case, and a suffix byte for byte in UTF-8.
