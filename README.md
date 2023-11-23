# second-diffusion

## hessian eigenvalues based on the score_sde_pytorch repo

### environment installation

```bash
cd [PATH]/second-diffusion/score_sde_pytorch
pip install -r requirements.txt
```

### hardware requirement

I have only ran this on an A100-80GB. It's possible to run into CUDA OOM error if we're not using a 80GB GPU instance. 

### pretrained checkpoints

petrained checkpoint should be in `https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf` and it's `checkpoint_26.pth`.

### command to get hessian eigenvalues based on the score_sde_pytorch

```bash
# accelerate_sampling
cd [PATH]/second-diffusion/score_sde_pytorch
python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/vp/ddpm/cifar10.py --eval_folder eval --mode sampling --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir
```
(also see `[PATH]/second-diffusion/scripts/submit_score_sde_pytorch_job.sh`)

### command to plot hessian eigenvalues based on the logs from `main.py`

```bash
cd [PATH]/second-diffusion/experiments
python visualize_score_sde_pytorch_samples.py
```
