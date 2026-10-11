# Model import, datasets and language

The work is specified by three design documents, ruled by the owner on 2026-10-11:

- [`MODEL_IMPORT.md`](../../design/MODEL_IMPORT.md): one import door, a model file and a
  `.sushi` weight file.
- [`DATASETS.md`](../../design/DATASETS.md): a batch with roles, format readers, collators,
  image decoding, segmentation.
- [`LANGUAGE.md`](../../design/LANGUAGE.md): the language tier, from the gather operation to
  the decision model and mixture of experts.

The owner ordered the code written before SushiBLAS follows SushiRuntime v0.4.0. No task of
this folder builds or runs a test; each is accepted on a syntax check of every unit it touches
and on the repository's checkers. The owner's build closes the work.
