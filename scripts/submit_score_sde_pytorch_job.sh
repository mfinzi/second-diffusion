#!/bin/bash


sbatch --job-name=score_sde_hessian \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --gres=gpu:a100:1 \
        --time=48:00:00 \
        --mem=72G \
        --mail-user=yk2516@nyu.edu \
        --error=/scratch/yk2516/slurm/precond_langevin/%j_%a_%N.err \
        --output=/scratch/yk2516/slurm/precond_langevin/%j_%a_%N.out \
        --wrap="singularity exec --nv --overlay $SCRATCH/singularity/overlay-25GB-500K-DiffusionPreconditioner.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c 'source /ext3/env.sh; conda activate base; cd /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch; python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/ve/ncsn/cifar10_yilun_sampling.py --eval_folder eval --mode sampling --use_pre_con False --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir'"
        # --wrap="singularity exec --nv --overlay $SCRATCH/singularity/overlay-25GB-500K-DiffusionPreconditioner.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c 'source /ext3/env.sh; conda activate base; cd /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch; python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/vp/ddpm/cifar10_accelerated_sampling.py --eval_folder eval --mode sampling --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir'"

# ? --use_pre_con True
# python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/ve/cifar10_ncsnpp_accelerated_sampling.py --eval_folder eval --mode sampling --use_pre_con True --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir

# ? --use_pre_con False
# python main.py --config /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/configs/ve/cifar10_ncsnpp_accelerated_sampling.py --eval_folder eval --mode sampling --use_pre_con True --workdir /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir

