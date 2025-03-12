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
    echo "Usage: $0 <pseudocount> [ts_mode]"
    echo "  pseudocount: Thompson sampling pseudocount value"
    echo "  ts_mode: Optional. Thompson sampling mode (category/embed/resample)"
    exit 1
fi

PSEUDOCOUNT=$1
TS_MODE=$2

# Set directory suffix based on pseudocount and ts_mode
if [ "$PSEUDOCOUNT" = "0" ] || [ -z "$TS_MODE" ] || [ "$TS_MODE" = "null" ]; then
    DIR_SUFFIX="base"
else
    DIR_SUFFIX="${TS_MODE}_${PSEUDOCOUNT}"
fi

echo "Pseudocount: $PSEUDOCOUNT"
echo "TS Mode: $TS_MODE"
echo "Directory suffix: $DIR_SUFFIX"

ulimit -n 64000
source ~/.bashrc
conda activate news
cd /iris/u/rypark/code/newsreclib
pwd

python newsreclib/eval.py experiment=dkn_mindsmall_pretrainedemb_celoss_bertsent \
    ckpt_path=/iris/u/rypark/code/newsreclib/logs/train/runs/dkn_mindsmall_pretrainedemb_celoss_bertsent_s42/2025-03-11_13-01-53/checkpoints/last.ckpt \
    logger=csv \
    model.ts_pseudocount=$PSEUDOCOUNT \
    model.ts_mode=$TS_MODE \
    hydra.run.dir=/iris/u/rypark/code/newsreclib/logs/eval/runs/dkn_mindsmall_pretrainedemb_celoss_bertsent/$DIR_SUFFIX
