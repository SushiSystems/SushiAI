# modules/vision/imaging/include/SushiAI/vision/imaging/polygon.hpp


## Point, Polygon

A polygon is the vertices of one object's outline in order, closed from the last vertex to the
first. An empty polygon means the object has none. The coordinates are fractions in a
`FileLabels` set and pixels in a `Labels` set.

## resampled_point_count, resample_polygon

The reference resamples every polygon of an image to 1000 points each time it loads the image,
or to the longest polygon's length plus one when that length is above 1000
(`data/dataset.py:416-422`). `resample_polygon` follows `resample_segments`
(`utils/ops.py:481-501`). With `m` the vertex count plus one (the first vertex appended) and
`n` the wanted count:

```
if vertex count == n:  unchanged
if m < n:   x = linspace(0, m - 1, n - m), then the integers 0 .. m - 1 merged in,
            each integer placed before the first x that is not below it
otherwise:  x = linspace(0, m - 1, n)
point(x) = vertex[j] + (vertex[j + 1] - vertex[j]) * (x - j),  j = floor(x)      (double)
```

`linspace(0, last, k)` is `i * (last / (k - 1))` for `i < k - 1` and `last` at the end, as
NumPy computes it. The result is cast to `float`. The resampled outline is the same closed
curve; it has more points so that the per-point clamp of the mosaic follows the canvas edge.

## map_polygon, clamp_polygon

`map_polygon` is `x * scale_x + offset_x`, `y * scale_y + offset_y` in `float`, multiply first.
Every transform moves a polygon through it. `clamp_polygon` clamps each vertex to a box.

## is_point_inside

Even-odd crossing count on a ray towards larger x, in `double`; a point on an edge or on a
vertex is inside. The reference calls `cv2.pointPolygonTest` here. OpenCV's source is not in
the Ultralytics package and was not read; the two can differ for a point within rounding of an
edge and for a self-intersecting outline.

## visible_polygon_box

The reference's `segment2box` (`utils/ops.py:78-124`), in `float`:

1. No vertex: four zeros.
2. Bounds inside [0, width] by [0, height]: the bounds.
3. Bounds wholly outside (`max < 0` or `min > extent` on an axis): four zeros.
4. Otherwise the bounds of three sets of points: the vertices inside the image; for each edge
   and each of the lines x = 0, x = width, y = 0, y = height, the point `start + t * delta`
   with `t = (bound - start) / delta`, kept when `t` is in [0, 1] and its other coordinate is
   inside the image; and each image corner that `is_point_inside` accepts. No point: four zeros.

An edge parallel to a line is skipped; the reference divides by zero there and discards the
result.
