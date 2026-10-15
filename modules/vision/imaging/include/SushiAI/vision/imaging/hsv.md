# modules/vision/imaging/include/SushiAI/vision/imaging/hsv.hpp


## HSVSettings, validate_hsv, plan_hsv

Gains 0.015, 0.7 and 0.4 (`cfg/default.yaml:115-117`). The reference draws
`np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]` (`data/augment.py:1517`), three values in
the order hue, saturation, value; `plan_hsv` draws the same three in the same order. The
reference makes no draw when all three gains are zero; `plan_hsv` still makes its three, which
changes nothing for another sample.

## is_hsv_identity

True when the plan's three gains are all zero. The reference skips the whole transform when the
three settings are zero (`data/augment.py:1514`); three zero settings always give such a plan,
and `apply_hsv` then copies the image instead of sending it through the hexcone round trip,
which would move a level by up to 4. A plan from non-zero settings is an identity only when all
three draws are exactly zero.

## make_hsv_tables

`data/augment.py:1518-1523`, in `double` as in the reference:

```
hue[x]        = trunc(floored_mod(x + plan.hue * 180, 180))
saturation[x] = trunc(clamp(x * (1 + plan.saturation), 0, 255))
value[x]      = trunc(clamp(x * (1 + plan.value), 0, 255))
saturation[0] = 0
```

`floored_mod(a, 180)` is `a - 180 * floor(a / 180)`, the result of `%` on NumPy floats.

## to_hsv, to_rgb

The reference converts with `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` and back
(`data/augment.py:1525-1527`). The `% 180` in its hue table shows that hue is 8-bit in steps of
two degrees, 0 to 179. The two functions below are the hexcone model in integers, written for
this module.

`to_hsv`, with `r`, `g`, `b` the levels, `v = max(r, g, b)` and `d = v - min(r, g, b)`:

```
value      = v
saturation = 0                       if v == 0
           = (255 * d + v / 2) / v   otherwise                (integer division)
hue        = 0                       if d == 0
otherwise:
    n = 30 * (g - b)              if v == r
      = 30 * (b - r) + 60 * d     else if v == g
      = 30 * (r - g) + 120 * d    otherwise
    if n < 0: n += 180 * d
    hue = (2 * n + d) / (2 * d)                                (integer division)
    if hue >= 180: hue -= 180
```

`to_rgb`, with `h`, `s`, `v` the levels, `i = h / 30` and `f = h - 30 * i`:

```
p = (v * (255 - s) + 127) / 255
q = (v * (7650 - s * f) + 3825) / 7650
t = (v * (7650 - s * (30 - f)) + 3825) / 7650                  (integer division, 7650 = 255 * 30)

i = 0: (v, t, p)    i = 1: (q, v, p)    i = 2: (p, v, t)
i = 3: (p, q, v)    i = 4: (t, p, v)    i = 5: (v, p, q)        as (red, green, blue)
```

The round trip is not the identity, because hue has 180 steps: (12, 200, 99) comes back as
(12, 200, 100).

## apply_hsv

Per pixel: `to_hsv`, the three tables, `to_rgb`. The image is copied unchanged when it does not
have three channels (`data/augment.py:1512-1513`) or when `is_hsv_identity(plan)`.
