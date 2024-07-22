#!/bin/bash

#SBATCH --job-name=mcmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.txt

python analysis/rl_models.py