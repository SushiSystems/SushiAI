# umbrella

## Owns

The single header that includes `<SushiAI/SushiAI.h>` and every public header of the vision
domain: 19 imaging, 17 detect, 5 blocks, 1 datasets and 1 registration header, 43 in all. The
module declares nothing and has no source of its own.

## Depends on

- `umbrella` of `layers` (`PUBLIC_DEPENDS`), which provides `<SushiAI/SushiAI.h>`.
- `vision/imaging`, `vision/detect`, `vision/blocks`, `vision/datasets` and
  `vision/registration` (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/vision.h>` | Nothing; it includes `<SushiAI/SushiAI.h>` and the 43 vision headers |
