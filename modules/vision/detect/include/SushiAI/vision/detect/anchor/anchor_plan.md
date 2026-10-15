# modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_plan.hpp


## plan_anchors

Takes the boxes, one stride per head and `AnchorPlanOptions`, and returns an `AnchorPlan`:

```
heads      = strides.size()
anchors    = per_head == 0 ? choose_anchor_count(boxes, heads, count).clustering.anchors
                           : cluster_anchors(boxes, count.clustering with count = per_head * heads)
layout     = assign_anchors_to_heads(anchors, strides, even split)
fit        = measure_anchor_fit(boxes, layout, fit options)
thresholds = derive_assignment_thresholds(fit)
```

`plan_anchors` holds none of the rules itself. Head `k` of the layout belongs to `strides[k]`, so
the strides may come in any order: under strides 32 and 16 head 0 takes the three largest of six
anchors. `trials` holds the counts `choose_anchor_count` tried, the refused successor last. With
`per_head` above 0 one clustering runs, `trials` is empty, and of `count` only `count.clustering`
is read. The count fixes the channel count of every head, so the plan is made before the model is
built.

The fit is measured on every box passed in. `cluster_anchors` leaves out a box with a side that
is not finite or not above `min_side`, and the plan does not report how many it left out; a
caller that needs the number calls `cluster_anchors` itself.

`plan_anchors` returns `InvalidDescription` for no stride, a negative `per_head`, a `per_head`
whose product with the head count does not fit in 64 bits, and whatever a step refuses as a bad
option or stride. It returns `Incompatible` for no box and for fewer distinct usable boxes than a
count that was tried. The steps run in the order above and the first failure is returned, so
boxes too few for a count are reported before a stride of zero.

The same boxes in the same order with the same strides and options give the same plan, bit for
bit on one build: the only random source is the seeded sequence of `cluster_anchors`.

`test_detect_anchor_plan.cpp` generates the Ladder and the Blob of SPEC 5.6 from seeds written in
the file. Six rungs under strides 16 and 32 give two heads of three anchors, a recall of 1 and
four trials. The test compares a plan with the four steps called one by one to 1e-9, and two
plans of the same arguments with a tolerance of 0.
