# modules/vision/imaging/include/SushiAI/vision/imaging/labels.hpp


## parse_label_text

The function takes text, not a path. A label file that does not exist is a background image in
the reference (`data/utils.py:483-484`); the caller passes empty text for it. The mapping from an
image path to its label path (`data/utils.py:143-144`) is the caller's and is not declared here.

Rows and fields (`data/utils.py:443`): the text is split into lines, a line with no field is
dropped, and a row's fields are separated by whitespace. A line ends at `\n`, `\r`, `\r\n`,
`\v`, `\f`, 0x1C, 0x1D or 0x1E; a field ends at a space, a tab or 0x1F. The non-ASCII
separators Python also honours (U+0085, U+00A0, U+2028, U+2029) are not separators here and
make the row fail as "not a number".

| Check | Reference |
| --- | --- |
| Every field parses, whole, as a finite decimal number | `np.array(lb, dtype=np.float32)`, `data/utils.py:451` |
| If any row has more than 6 fields, the file is a polygon file | `data/utils.py:446` |
| In a polygon file every row has an odd field count of at least 7 | `data/utils.py:447`, `449` |
| Otherwise every row has exactly 5 fields | `data/utils.py:457` |
| Each of the four box values is at most 1.01 | `data/utils.py:460` |
| The class and each box value are at least -0.01 | `data/utils.py:461` |
| The class has no fractional part | `data/utils.py:462` |
| The class is below `class_count` | `data/utils.py:465-466` |

A polygon row is `class x1 y1 x2 y2 ...`. Its box is the bounds of its points,
`x_center = (min_x + max_x) / 2` and `width = max_x - min_x` in `float`
(`utils/ops.py:465-478`, `257-260`), and its points are kept, in file order, as the object's
polygon (`data/utils.py:446-450`). The coordinate checks apply to the box, not to the points,
as in the reference. Values inside the tolerance are stored as they are; nothing is clipped.
The points are not resampled here: `resample_polygons` does that on each load, because 1000
points for every object of a dataset would not fit in memory.

Any failed check throws `Error`. The reference drops the image and counts it as corrupt
(`data/utils.py:492-495`); the caller that builds a dataset decides the same here by catching.

Three rules are narrower or simpler than the reference:

- The reference accepts a polygon row of one point (3 fields) in a polygon file. This parser
  requires three points.
- A nonzero field whose magnitude is below the least normal `double` is rejected. The reference
  accepts it as a subnormal or as zero. The rule keeps the outcome the same on every standard
  library; the message differs ("below the smallest normal number" where `std::from_chars`
  returns the subnormal, "not a number" where it reports out of range).
- Duplicates. The reference removes repeated rows (`data/utils.py:470-475`) and, when it removes
  any, returns the survivors in `np.unique`'s sorted order. This parser keeps the first
  occurrence of each row and the file's order. A box row repeats another when all five values
  are equal; a polygon row, when the class and every point are equal. Only the order of the
  objects inside one image differs.
