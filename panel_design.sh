#!/bin/bash
#SBATCH --job-name=panel-design
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

# BARseq2 gene-panel selection from the Allen 2020 reference.
# The one-time streaming pass is cached in results/panel_cache; selection reuses it.
# (panel_design.py is self-contained, so we run it directly with the env interpreter
#  rather than importing the iss_analysis package, whose __init__ pulls in iss_preprocess.)

cd /camp/home/znamenp/home/users/znamenp/code/iss-analysis
PY=/camp/home/znamenp/.conda/envs/iss-analysis/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

$PY iss_analysis/panel_design.py \
    --datapath /camp/home/znamenp/home/shared/resources/allen2020/ \
    --cache_dir results/panel_cache \
    --out_dir results/panel \
    --subsample_n 150000 \
    --efficiency 0.1 \
    --n_select 400 \
    --lam 0.5 \
    --min_max_subclass_mean 1 \
    --min_tau 0.3 \
    --max_candidates 2000 \
    --n_accuracy_cells 20000 \
    --overlap_cells 6000
