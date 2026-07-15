#!/bin/bash
#SBATCH --job-name=ensemble_sweep
#SBATCH --output=logs/shapley%j.out
#SBATCH --error=logs/shapley%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mpcg.p
#SBATCH --gres=gpu:1

module load Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/mds-env/bin/activate

SCRIPT_DIR=/user/gaad2403/MDS-ED/key/Final/mortality
mkdir -p $SCRIPT_DIR/logs $SCRIPT_DIR/results/csv $SCRIPT_DIR/results/png

echo "Start: $(date)"
python $SCRIPT_DIR/shapley.py
echo "Done: $(date)"
