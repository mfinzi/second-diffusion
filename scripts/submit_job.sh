#!/bin/bash


sbatch --job-name=langevin \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --gres=gpu:1 \
        --time=48:00:00 \
        --mem=72G \
        --mail-user=yk2516@nyu.edu \
        --error=/scratch/yk2516/slurm/precond_langevin/%j_%a_%N.err \
        --output=/scratch/yk2516/slurm/precond_langevin/%j_%a_%N.out \
        --wrap="singularity exec --nv --overlay $SCRATCH/singularity/overlay-25GB-500K-DiffusionPreconditioner.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c 'source /ext3/env.sh; conda activate base; cd /scratch/yk2516/repos/diffusion_model/second-diffusion/diffusion; python accelerated_sampling_yilun_draft.py'"



 