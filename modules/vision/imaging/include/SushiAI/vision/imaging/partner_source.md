# modules/vision/imaging/include/SushiAI/vision/imaging: partner_source.hpp, dataset_partners.hpp, recent_partners.hpp


## PartnerSource

The seam with one job: `choose(index, random)` returns the three partner indices of one sample
from exactly three draws. `draw_mosaic` calls it once per sample, before the gate, so a source
that keeps state sees every sample of an open epoch. A training pipeline owns its source
through a `std::unique_ptr`.

## RecentPartners, the default

The reference's rule when images are not cached in RAM, which is its default (`cache: False`,
`cfg/default.yaml:19`; `data/augment.py:482`, `499-500`). The reference keeps a list of the
indices it loaded most recently and draws partners from it with `random.choices`.

- Length (`data/base.py:168`): `recent_capacity = min(dataset size, 8 * batch size, 1000)`.
- Update (`data/base.py:306-312`): an index not in the buffer is appended; then, when the
  buffer holds more than one entry and at least `capacity` of them, the oldest leaves. A full
  buffer therefore holds `capacity - 1` indices, and one index when the capacity is 1 or 2.
- An index already in the buffer is not appended again and does not move.
- Draw: three times `recent[random.index(recent.size())]`. The sample's own index is in the
  buffer by then, so the first sample of a pipeline has itself as all three partners.

`TrainingPipeline(source, settings)` builds this source from `source.size()` and
`settings.batch_size` (16, `cfg/default.yaml:15`).

## DatasetPartners

The reference's rule under a RAM cache (`data/augment.py:501-502`): three times
`random.index(dataset size)`. It keeps no state.
