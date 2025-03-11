#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=240:00:00
#SBATCH --job-name=train
#SBATCH --output slurm/%j.out

ulimit -n 64000
source ~/.bashrc
conda activate news
cd /iris/u/rypark/code/newsreclib
pwd

python newsreclib/train.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent
