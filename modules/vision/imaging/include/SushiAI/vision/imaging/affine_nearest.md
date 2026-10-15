# modules/vision/imaging/include/SushiAI/vision/imaging/affine_nearest.hpp


## apply_affine_nearest

The warp of `apply_affine` with one tap in place of four. Output pixel `(x, y)` reads the source
pixel at `((x - m[2]) / m[0], (y - m[5]) / m[4])`, each coordinate computed in `double` and
rounded half up, `floor(position + 0.5)`. A coordinate that rounds outside the source gives
`plan.border_value`. No two source values are blended, so an image of class indices holds only
indices of its source and the border value. Every channel of a pixel moves together.

It takes the `AffinePlan` that `plan_affine` made for the image, so a mask warped with it lies
under the image `apply_affine` warped: both read the same source position, one blends its four
neighbours and the other takes the nearest. The caller sets `border_value` to the byte that
names no class.

It returns a `Status` and throws nothing.

| Code | When |
| --- | --- |
| `InvalidDescription` | A source of no pixels or of a size or channel count outside the bounds; a source that shares pixels with the destination; an output size outside the bounds; matrix scales that are not finite, not above zero or not `plan.scale`; an off-diagonal term; a translation that is not finite |
| `Incompatible` | A source of another size than `plan.source` |
| `OutOfMemory` | An allocation fails |
