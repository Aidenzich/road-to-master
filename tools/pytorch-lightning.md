# Pytorch Lightning
| Properties  | Data |
|-|-|
| Created | 2022-12-19 |
| Updated | 2022-12-19 |
| Author | @Aiden |
| Tags | #study |

- [package](https://github.com/PyTorchLightning/pytorch-lightning)
## Benefit of Pytorch Lightning
### Clear Training, Validation and Inference process
```python
    def training_step(self, batch, batch_idx) -> Tensor:
        ...
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        ...
        self.log("val_acc", torch.FloatTensor([acc]))

    def test_step(self, batch, batch_idx) -> None:
        ...
        self.log("test_acc", torch.FloatTensor([acc]))
```

### 3 Ways to speed up training with Pytorch Lightning
- Mixed Precision
- Multi-GPU Training
- Easy use EarlyStopping

#### Mixed Precision
| Type | Meaning | Examples |
|-|-|-|
| Lower precision | Use less memory, easier to train and deploy massive neural network | `16-bit` |
| Higher precision | Can be used for particularly sensitive use-cases. | `64-bit` |

```python
from pytorch_lightning import Trainer

trainer = Trainer(precision=16)
```

#### Multi-GPU Training
Devices will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`, based on the accelerator type.
```python
from pytorch_lightning import Trainer

trainer = Trainer(devices=-1)  # or the number of GPUs available
```
- Availble Value
    | Value | Meaning |
    |-|-|
    | `-1` | Use all available GPUs |
    | `n` | Use `n` number available GPUs |
    | [`id1`, `id2`...] | Use specific GPUs by device id |

#### EarlyStopping
You can easily define which data you want to monitor, like `loss`, `acc`.
```python
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer

es = EarlyStopping(
    monitor="val_loss",
    stopping_threshold=1e-4,
    divergence_threshold=6.0
)

trainer = Trainer(callbacks=[es])
```
