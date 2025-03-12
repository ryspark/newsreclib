#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=240:00:00
#SBATCH --job-name=eval
#SBATCH --output slurm/%j.out

ulimit -n 64000
source ~/.bashrc
conda activate news
cd /iris/u/rypark/code/newsreclib
pwd

python newsreclib/eval.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-01_16-53-36/checkpoints/last.ckpt logger=csv

#python newsreclib/eval.py experiment=tanr_mindsmall_pretrainedemb_celoss_bertsent_s42 ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/tanr_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-01_17-27-43/checkpoints/last.ckpt logger=csv

#python newsreclib/eval.py experiment=sentidebias_mindsmall_pretrainedemb_celoss_bertsent_s42 ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/sentidebias_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-01_17-27-43/checkpoints/last.ckpt logger=csv

python newsreclib/eval.py experiment=dkn_mindsmall_pretrainedemb_celoss_bertsent ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/dkn_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-11_13-01-53 logger=csv
