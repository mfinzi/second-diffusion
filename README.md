# second-diffusion

## hessian eigenvalues based on the score_sde_pytorch repo

### environment installation

To install the environment, run

```bash
conda env create -f environment.yml
```

or
```bash
cd [PATH]/second-diffusion
pip install -r requirements.txt

cd [PATH]/second-diffusion/score_sde_pytorch
pip install -r requirements.txt
```

If you are on Greene, you can also just copy the overlay image `/scratch/yk2516/singularity/overlay-25GB-500K-DiffusionPreconditioner.ext3` since it already have all environment set up.

### hardware requirement

I have only ran this on an A100-80GB. It's possible to run into CUDA OOM error if we're not using a 80GB GPU instance.

### pretrained checkpoints

petrained checkpoint should be in `https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf` and it's `checkpoint_26.pth`. This checkpoint should be put under `[PATH]/second-diffusion/score_sde_pytorch/work_dir/checkpoints`. Here is an example `/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/checkpoints/checkpoint_26.pth`

### command to get hessian eigenvalues based on the score_sde_pytorch

```bash
# accelerate_sampling
cd [PATH]/second-diffusion/score_sde_pytorch
python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/vp/ddpm/cifar10_accelerated_sampling.py --eval_folder eval --mode sampling --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir
```
(also see `[PATH]/second-diffusion/scripts/submit_score_sde_pytorch_job.sh`)

### how are output files from jobs saved?

All output file from running the above `main.py` command should be saved in the form of `[PATH]/second-diffusion/score_sde_pytorch/work_dir/eval/ckpt_[CKPT_NUM]/[DATE]/[TIME]`. Here is an example of the output:

```bash
/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/eval/ckpt_26/2023-11-24/12-42-02
```

### command to plot hessian eigenvalues based on the logs from `main.py`

```bash
cd [PATH]/second-diffusion/experiments
python visualize_score_sde_pytorch_samples.py --sample_storage_path /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/eval/ckpt_26/2023-11-24/12-42-02
```
