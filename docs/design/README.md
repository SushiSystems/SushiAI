# Design map

Each topic maps to one design document.

## Topics

<!-- topics:begin -->
| Topic | State | About |
| --- | --- | --- |
| [DATASETS](DATASETS/README.md) | Open | How SushiAI reads training data for more than one kind of task, from the layouts data is commonly kept in, without a class per named dataset. |
| [DOMAIN_MODULES](DOMAIN_MODULES/README.md) | Open | How SushiAI keeps what belongs to one field of use (vision today; language, physics and reinforcement learning later) apart from the library every field shares, so that a field can be... |
| [GRAPH_CAPABILITIES](GRAPH_CAPABILITIES/README.md) | Open | The changes below SushiAI's layers tier that lift the limits found while the language, dataset, model-import and Darknet work was written in October 2026. |
| [LANGUAGE](LANGUAGE/README.md) | Open | How SushiAI trains and runs a decoder-only transformer on text, and an encoder that answers typed questions in one forward pass: the second domain after vision, laid out as... |
| [LAYERED_EXECUTION](LAYERED_EXECUTION/README.md) | Open | SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and it trains one model over the owner's five machines through SushiRuntime's distributed... |
| [MODEL_IMPORT](MODEL_IMPORT/README.md) | Open | How a model of another framework (Darknet today; PyTorch weight files, ONNX and Caffe by the format engine below) enters SushiAI once, through one door, and is trained and run afterwards... |
| [PROFILING](PROFILING/README.md) | Open | How SushiAI measures where a training step and an inference batch spend their time, through SushiProfiler alone, down to the IR node each kernel belongs to. |
| [PROGRAMME](PROGRAMME/README.md) | Open | What SushiAI is being built towards, in what order, and the owner's rulings that fix the shape of the work. |
<!-- topics:end -->
