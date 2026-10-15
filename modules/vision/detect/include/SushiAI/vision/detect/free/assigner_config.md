# modules/vision/detect/include/SushiAI/vision/detect/free/assigner_config.hpp


## AssignerConfig

The member defaults are the constructor defaults of `TaskAlignedAssigner`
(`U/utils/tal.py:33-39`): `topk = 13`, `num_classes = 80`, `alpha = 1.0`, `beta = 6.0`,
`eps = 1e-9`, `topk2 = topk`. `stride_val` is the second stride, or the only one
(`U/utils/tal.py:58-59`), which is 16 for the default strides 8, 16 and 32.

## one_to_many_assigner_config, one_to_one_assigner_config

The detection loss builds both assigners with `alpha=0.5, beta=6.0` and the model's strides
(`U/utils/loss.py:374-381`), and `E2ELoss` chooses the candidate counts:

```
U/utils/loss.py:1327  self.one2many = loss_fn(model, tal_topk=10)
U/utils/loss.py:1328  self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
```

`MANIFEST.json` records the same values as read back from the reference's assigner objects.
