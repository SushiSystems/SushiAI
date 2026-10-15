# modules/vision/datasets/include/SushiAI/vision/datasets/cifar.hpp


A record is its label bytes and then 3072 pixel bytes: a 32 by 32 red plane, a green one and a
blue one, each row-major.

| `CIFARVariant` | Label bytes | Label taken | Classes | `train` | `test` |
| --- | --- | --- | --- | --- | --- |
| `CIFAR10` | 1 | Byte 0 | 10 | `data_batch_1.bin` to `data_batch_5.bin` | `test_batch.bin` |
| `CIFAR100Fine` | 2 | Byte 1 | 100 | `train.bin` | `test.bin` |
| `CIFAR100Coarse` | 2 | Byte 0 | 20 | `train.bin` | `test.bin` |

The rows are `K_FORMATS` in `source/cifar.cpp`. The reader seeks to the label byte of each record
and reads it; it reads no pixel. The label that is not taken is not checked.

| Code | When |
| --- | --- |
| `InvalidDescription` | `split` is neither `train` nor `test` |
| `FileNotFound` | A file of the split is absent |
| `InvalidFile` | A file is empty or not a whole number of records; a label is at or above the class count, and the message gives the file and the one-based record |
| `OutOfMemory` | An allocation fails |
