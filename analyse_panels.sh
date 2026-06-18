#!/bin/bash
#SBATCH --job-name=panel-figs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

# Full panel_plots analysis suite for ONE selection output dir.
#     sbatch --export=ALL,PANEL_DIR=results/panel_subclass_e0.85 analyse_panels.sh

cd /camp/home/znamenp/home/users/znamenp/code/iss-analysis
PY=/camp/home/znamenp/.conda/envs/iss-analysis/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "analysing $PANEL_DIR"
$PY results/make_all_figures.py $PANEL_DIR
echo "DONE $PANEL_DIR"
