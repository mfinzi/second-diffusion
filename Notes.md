## Commands

#### Getting the checkpoint
For DDPM you can download `checkpoint_26.pth` from [here](https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf).

### Quick run
```shell
py main.py --config configs/vp/ddpm/cifar10_accelerated_sampling.py --eval_folder eval --mode sampling --workdir .
```
