# GGNN

## Environment
* Python >= 3.6
* cuda == 10.2
* Pytorch==1.5.1
* PyG: torch-geometric==1.5.0


## Result Reproduction
Run the following command:
```
    # cpu
    bash run.sh cpu msl
    # gpu
    bash run.sh <gpu_id> msl
```

```bash
epoch (0 / 100) (Loss:0.08383194, ACU_loss:5.86823592)
epoch (1 / 100) (Loss:0.05875191, ACU_loss:4.11263351)
epoch (2 / 100) (Loss:0.05632888, ACU_loss:3.94302161)
epoch (3 / 100) (Loss:0.05274593, ACU_loss:3.69221499)
epoch (4 / 100) (Loss:0.04955142, ACU_loss:3.46859971)
epoch (5 / 100) (Loss:0.04796321, ACU_loss:3.35742461)
epoch (6 / 100) (Loss:0.04487637, ACU_loss:3.14134582)
epoch (7 / 100) (Loss:0.04429099, ACU_loss:3.10036950)
...

==================** Result **====================

F1 score: 0.947151529004282
precision: 0.901399660359049
recall: 0.9977961432506887
```
