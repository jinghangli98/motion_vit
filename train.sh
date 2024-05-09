#!/bin/bash

#SBATCH --job-name="tse-motion"
#SBATCH --partition=gpu
#SBATCH --output=tse-motion.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --time=99:00:00
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --account=haizenstein

source activate pytorch
python3 ./train.py
