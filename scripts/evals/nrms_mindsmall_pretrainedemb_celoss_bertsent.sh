#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=240:00:00
#SBATCH --job-name=eval
#SBATCH --output slurm/%j.out

# Check if at least one argument is provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <pseudocount> [use_icl]"
    echo "  pseudocount: Thompson sampling pseudocount value"
    echo "  use_icl: Optional. If specified and non-zero, enables ICL mode"
    exit 1
fi

PSEUDOCOUNT=$1
USE_ICL=false
DIR_PREFIX="prior"

# If second argument is provided and non-zero, enable ICL
if [ $# -eq 2 ] && [ $2 -ne 0 ]; then
    USE_ICL=true
    DIR_PREFIX="icl"
fi

# Set directory suffix based on pseudocount
if [ "$PSEUDOCOUNT" = "0" ]; then
    DIR_SUFFIX="base"
else
    DIR_SUFFIX="${DIR_PREFIX}_$PSEUDOCOUNT"
fi

echo $PSEUDOCOUNT
echo $DIR_SUFFIX
echo "ICL mode: $USE_ICL"

ulimit -n 64000
source ~/.bashrc
conda activate news
cd /iris/u/rypark/code/newsreclib
pwd

python newsreclib/eval.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent \
    ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-01_16-53-36/checkpoints/last.ckpt \
    logger=csv \
    model.ts_pseudocount=$PSEUDOCOUNT \
    model.ts_icl=$USE_ICL \
    hydra.run.dir=/iris/u/rypark/code/newsreclib/logs/eval/runs/nrms_mindsmall_pretrainedemb_celoss_bertsent/$DIR_SUFFIX
