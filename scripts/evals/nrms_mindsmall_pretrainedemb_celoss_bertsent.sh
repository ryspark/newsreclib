#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=240:00:00
#SBATCH --job-name=eval
#SBATCH --output slurm/%j.out

# Check if pseudocount argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <pseudocount>"
    exit 1
fi

PSEUDOCOUNT=$1

# Set directory suffix based on pseudocount
if [ "$PSEUDOCOUNT" = "0" ]; then
    DIR_SUFFIX="base"
else
    DIR_SUFFIX="prior_$PSEUDOCOUNT"
fi

echo $PSEUDOCOUNT
echo $DIR_SUFFIX

ulimit -n 64000
source ~/.bashrc
conda activate news
cd /iris/u/rypark/code/newsreclib
pwd

python newsreclib/eval.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent \
    ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-01_16-53-36 \
    logger=csv \
    model.ts_pseudocount=$PSEUDOCOUNT \
    hydra.run.dir=/iris/u/rypark/code/newsreclib/logs/eval/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent_s42/$DIR_SUFFIX 
