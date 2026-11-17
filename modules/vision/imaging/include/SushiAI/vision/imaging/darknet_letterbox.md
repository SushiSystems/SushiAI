# modules/vision/imaging/include/SushiAI/vision/imaging/darknet_letterbox.hpp


## plan_darknet_letterbox

Darknet's `letterbox_image` (`image.c` of the Darknet repository), with `w`, `h` the network input and `im` the image:

```
if ((float) w / im.w < (float) h / im.h) { new_w = w;  new_h = (im.h * w) / im.w; }
else                                     { new_h = h;  new_w = (im.w * h) / im.h; }
dx = (w - new_w) / 2        dy = (h - new_h) / 2
```

Every division of the second and third lines is integer division, so the sides and the offsets
truncate. `plan_letterbox` rounds the sides half to even, splits an odd padding the other way and
never scales up unless asked, which is the Ultralytics rule; the two plans differ on most sizes.

## apply_darknet_letterbox

The image is resized to `new_w` by `new_h` with `resize_darknet`, the corner-to-corner kernel of
Darknet, and embedded at `(dx, dy)` in a target filled with 0.5. In 8 bits that is
`K_DARKNET_PADDING`, 128; Darknet's float 127.5 has no 8-bit level, and 128 is its round half up.

Labels move by `x * new_w / im.w + dx` and `y * new_h / im.h + dy`. The two scales differ by
less than one pixel's share, since both sides truncate, so a box keeps its aspect up to that.
