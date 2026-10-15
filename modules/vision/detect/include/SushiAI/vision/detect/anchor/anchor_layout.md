# modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_layout.hpp


## AnchorLayout

`anchors` lists every anchor of the model. `head_anchors[k]` lists the anchors head `k` uses, as
indices into `anchors`; entry `a` of that list is the head's anchor slot `a`, which fixes the
order of its channels and of its decoded rows.

## validate_anchor_layout

Returns `InvalidDescription` for a layout without anchors or without heads, an anchor whose side
is not finite and above zero, a head without anchors, and an index below 0 or past the anchors.
The message names the anchor, head and slot. Two heads may share an anchor, and an anchor may
belong to no head: a layout read from a file can have both, and `validate_anchor_layout` accepts
it. A head that lists one anchor twice is not refused.
