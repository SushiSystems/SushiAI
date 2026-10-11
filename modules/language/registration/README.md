# registration

## Owns

The three functions that fill the registries with language rows. Each returns a `Status`:
success, or the first failure of the registry's `add()`. The header forward-declares the three
registries; their declarations live in `node_registry.hpp`, `dataset_registry.hpp` and
`objective_registry.hpp`.

| Function | Adds | Factory |
| --- | --- | --- |
| `register_nodes` | `position_embedding`, `causal_self_attention`, `feed_forward`, `transformer_block`, `moe_feed_forward`, as node rows, in that order | The `make_*_node` function of each block in `language/blocks` |
| `register_nodes` | `language_head`, as a linked node row | `Blocks::make_language_head_node`, which reads the rows above for `tie` |
| `register_datasets` | `tokens` | `Datasets::make_tokens_dataset` |
| `register_datasets` | `prompt` | `Datasets::make_prompt_dataset` |
| `register_objectives` | `next_token` | `Training::make_next_token_objective` |
| `register_objectives` | `expert_balance` | `Training::make_expert_balance_objective` |

`embedding`, `layernorm` and `dropout` are rows of `nn`, which `NN::register_layers` adds.

This module is the one place that adds the two objective rows; `language/training` declares
the objectives and their factories and knows no registry. It is also the one module that
sees both `language/blocks` and `language/training`, so its test compares the port names of
a block with experts with the prefixes `expert_balance` looks for.

## A decoder with experts

A `transformer_block` row that states `experts` has three ports, and the model publishes the
two beside its result by name:

```json
{
    "format_version": 2,
    "input": { "shape": [256], "dtype": "int32" },
    "layers": [
        { "type": "embedding", "num_embeddings": 4096, "embedding_dim": 384 },
        { "type": "position_embedding" },
        { "type": "transformer_block", "heads": 6, "experts": 4, "top_k": 2 },
        { "type": "language_head", "tie": 0 }
    ],
    "outputs": [
        { "node": 3 },
        { "node": 2, "port": "balance", "name": "balance2" },
        { "node": 2, "port": "usage", "name": "usage2" }
    ]
}
```

Its training file adds the balance to the fit through the `sum` row of `train`:

```json
"objective": {
    "type": "sum",
    "terms": [
        { "type": "next_token", "output": "output" },
        { "type": "expert_balance", "weight": 0.01 }
    ]
}
```

The weight is an example and not a measured value. `sum` refuses a weight of 0, so a run
without the balance states `next_token` alone, with its `output` key.

## Depends on

- `core`, `data`, `nn` and `train` (`PUBLIC_DEPENDS`).
- `language/blocks`, `language/datasets` and `language/training` (`PRIVATE_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/registration.hpp>` | `register_nodes`, `register_datasets`, `register_objectives` |

## A decoder as a model file

With the rows of `nn` and of this module a decoder is a model file:

```json
{
    "format_version": 2,
    "input": { "shape": [256], "dtype": "int32" },
    "layers": [
        { "type": "embedding", "num_embeddings": 4096, "embedding_dim": 384 },
        { "type": "position_embedding" },
        { "type": "transformer_block", "heads": 6 },
        { "type": "language_head", "tie": 0 }
    ]
}
```

`applications/examples/tiny_decoder/` holds one of five blocks with its training file.

Source: `docs/design/LANGUAGE.md`, and `docs/design/DOMAIN_MODULES.md` for the shape of a
registration module.
