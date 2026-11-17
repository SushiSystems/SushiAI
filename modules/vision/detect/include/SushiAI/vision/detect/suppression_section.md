# suppression_section.hpp

`SuppressionSection` holds what the `suppression` object of a task section states: the
criterion, the threshold and the distance exponent. `read_suppression_section` reads the member
from a task object and `write_suppression_section` writes it.

`criterion` is `"iou"` or `"distance_iou"`, `threshold` is a number from 0 to 1, and
`distance_exponent` is a finite number that `distance_iou` requires and `iou`
refuses. Every refusal is `InvalidDescription`; it starts with the name the caller gives the task
and names the key.

`SuppressionKeys` selects the members the object carries. `WithThreshold` is the default, and
`WithoutThreshold` is the anchor task's form: the object holds `criterion` and
`distance_exponent` only, the reader refuses `threshold` as an unknown key, and the writer
leaves it out.

`yolox/yolox_task.hpp`, `free/free_task.cpp` and `anchor/anchor_task.cpp` read and write their
suppression through this header; none holds its own criterion table.
