# umbrella

## Owns

The single header that includes `<SushiAI/SushiAI.hpp>` and every public header of the vision
domain: 19 imaging, 17 detect, 5 blocks, 1 datasets and 1 registration header, 43 in all. The
module declares nothing and has no source of its own.

## Depends on

- `umbrella` of `layers` (`PUBLIC_DEPENDS`), which provides `<SushiAI/SushiAI.hpp>`.
- `vision/imaging`, `vision/codecs`, `vision/detect`, `vision/segment`, `vision/blocks`,
  `vision/datasets`, `vision/importers`, `vision/training` and `vision/registration`
  (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision.hpp>` | Nothing; it includes `<SushiAI/SushiAI.hpp>` and the 43 vision headers |
