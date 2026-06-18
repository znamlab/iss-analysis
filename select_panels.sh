#!/bin/bash
#SBATCH --job-name=panel-select
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL

# BARseq panel selection for ONE level (subclass or cluster). Reference is library-size
# normalised internally (per-cell depth). Reuses the cached streaming pass.
#
# Submit one job per (level, min_expr), e.g.:
#     sbatch --export=ALL,LEVEL=subclass,MIN_EXPR=0.85 select_panels.sh
#     sbatch --export=ALL,LEVEL=cluster,MIN_EXPR=1.5  select_panels.sh
# Override any of: LEVEL, MIN_EXPR, MAXC, NSEL, EFF.

cd /camp/home/znamenp/home/users/znamenp/code/iss-analysis
PY=/camp/home/znamenp/.conda/envs/iss-analysis/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

LEVEL=${LEVEL:-subclass}        # subclass (pure 16-way) or cluster (pure ~215-way)
MIN_EXPR=${MIN_EXPR:-0.85}      # gene peak-subclass mean (normalised); 0.85 ~ 2000 candidates
ACC_OBJ=${ACC_OBJ:-soft_acc}    # greedy score: soft_acc | logpost | accuracy
MAXC=${MAXC:-2000}             # greedy candidate-pool cap (top by marginal MI at the level)
NSEL=${NSEL:-400}             # genes to select
EFF=${EFF:-0.1}              # simulated BARseq efficiency

# objective tag in dir name (legacy hard-accuracy keeps the un-suffixed name)
[ "$ACC_OBJ" = "accuracy" ] && SUFFIX="" || SUFFIX="_${ACC_OBJ}"
OUT=results/panel_${LEVEL}_e${MIN_EXPR}${SUFFIX}
echo "LEVEL=$LEVEL MIN_EXPR=$MIN_EXPR ACC_OBJ=$ACC_OBJ MAXC=$MAXC NSEL=$NSEL EFF=$EFF -> $OUT"

$PY iss_analysis/panel_design.py \
    --cache_dir results/panel_cache --out_dir $OUT \
    --level $LEVEL --accuracy_objective $ACC_OBJ --efficiency $EFF --n_select $NSEL \
    --min_max_subclass_mean $MIN_EXPR --min_tau 0.3 --max_candidates $MAXC

echo "DONE: $OUT"
