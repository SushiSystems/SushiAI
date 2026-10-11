# Design map

Each topic maps to one design document.

| Topic | Document | Status |
| --- | --- | --- |
| Where SushiAI is going: the target, the owner's rulings and the stages | [`PROGRAMME.md`](PROGRAMME.md) | Open — stage 1 of 7 built and passing |
| How a field of use (vision, later language, physics, reinforcement) is kept apart: tiers, the shape of a domain, registration and targets | [`DOMAIN_MODULES.md`](DOMAIN_MODULES.md) | Open — built; the training parity build is pending |
| How a model of another framework enters once and is trained from SushiAI's own files: the import step, the model file's task section | [`MODEL_IMPORT.md`](MODEL_IMPORT.md) | Open — phase 0 of 4 |
| How training data is read for several tasks: a batch of roles, format readers, collators, image decoding, segmentation and the binary task | [`DATASETS.md`](DATASETS.md) | Open — phase 0 of 7 |
| Which changes below the layers tier lift the limits of the October 2026 work: the plan for SushiRuntime, SushiBLAS and the SushiAI graph | [`GRAPH_CAPABILITIES.md`](GRAPH_CAPABILITIES.md) | Open — phase 0 of 4; plan written and its decisions ruled, nothing implemented |
| How a decoder-only transformer trains and runs on text: the language tier, the operations the core tier gains, the tokenizer, generation, the decision model, mixture of experts | [`LANGUAGE.md`](LANGUAGE.md) | Open — phases 1 to 5 and 7 of 7 written, not yet built |
| Building SushiBLAS and SushiAI up to the MLP showcase: the ST, SB and AI tracks | [`ROADMAP.md`](ROADMAP.md) | Shipped — superseded by `PROGRAMME.md` |
