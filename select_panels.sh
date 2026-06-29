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

LEVEL=${LEVEL:-subclass}        # subclass | cluster | hierarchical
MIN_EXPR=${MIN_EXPR:-0.85}      # gene peak-subclass mean (normalised); 0.85 ~ 2000 candidates
MIN_TAU=${MIN_TAU:-0.3}         # specificity floor (tau); 0 disables the specificity filter
LAM=${LAM:-0.5}                # cluster weight for hierarchical objective
ACC_OBJ=${ACC_OBJ:-soft_acc}    # greedy score: soft_acc | logpost | accuracy
CACHE=${CACHE:-results/panel_cache}   # streaming-pass cache (e.g. results/panel_cache_visp)
PREFIX=${PREFIX:-}             # output-name prefix, e.g. "visp_" for VISp-only training
MAXC=${MAXC:-2000}             # greedy candidate-pool cap (top by marginal MI at the level)
NSEL=${NSEL:-400}             # genes to select
EFF=${EFF:-0.1}              # simulated BARseq efficiency

# Step-wise (seed-chained) selection: STEPWISE=1, optional STAGES override.
#     sbatch --export=ALL,STEPWISE=1,MIN_EXPR=3,MIN_TAU=0.4 select_panels.sh
# Custom starting/seed genes (override the conventional-marker seed) via SEED_FILE (a CSV;
# gene column = SEED_COL, default 2 for codebook GII,barcode,gene) and/or SEED_GENES (comma list):
#     sbatch --export=ALL,STEPWISE=1,PREFIX=cb83_,STAGES=subclass:200,cluster:100,overlap:100,\
#            SEED_FILE=.../codebook_83gene_pool.csv,SEED_GENES=Gad1,Slc17a7,Vip,Sst select_panels.sh
STEPWISE=${STEPWISE:-0}
STAGES=${STAGES:-subclass:200,cluster:100,overlap:100}
SEED_FILE=${SEED_FILE:-}
SEED_COL=${SEED_COL:-2}
SEED_GENES=${SEED_GENES:-}
if [ "$STEPWISE" = "1" ]; then
    # "e" in the dir name is min_expr; tag non-default efficiency separately to avoid collisions
    EFFTAG=""; [ "$EFF" != "0.1" ] && EFFTAG="_eff${EFF}"
    OUT=results/panel_${PREFIX}stepwise_e${MIN_EXPR}_t${MIN_TAU}${EFFTAG}
    SEED_ARGS=""
    [ -n "$SEED_FILE" ]  && SEED_ARGS="$SEED_ARGS --seed_genes_file $SEED_FILE --seed_genes_col $SEED_COL"
    [ -n "$SEED_GENES" ] && SEED_ARGS="$SEED_ARGS --seed_genes $SEED_GENES"
    echo "STEPWISE CACHE=$CACHE STAGES=$STAGES MIN_EXPR=$MIN_EXPR MIN_TAU=$MIN_TAU ACC_OBJ=$ACC_OBJ MAXC=$MAXC SEED_FILE=$SEED_FILE SEED_GENES=$SEED_GENES -> $OUT"
    $PY iss_analysis/panel_design.py \
        --cache_dir $CACHE --out_dir $OUT \
        --stepwise --stages "$STAGES" --accuracy_objective $ACC_OBJ --efficiency $EFF \
        --min_max_subclass_mean $MIN_EXPR --min_tau $MIN_TAU --max_candidates $MAXC $SEED_ARGS
    echo "DONE: $OUT"
    exit 0
fi

# objective tag in dir name (legacy hard-accuracy keeps the un-suffixed name)
[ "$ACC_OBJ" = "accuracy" ] && SUFFIX="" || SUFFIX="_${ACC_OBJ}"
OUT=results/panel_${PREFIX}${LEVEL}_e${MIN_EXPR}_t${MIN_TAU}${SUFFIX}
echo "CACHE=$CACHE LEVEL=$LEVEL MIN_EXPR=$MIN_EXPR MIN_TAU=$MIN_TAU LAM=$LAM ACC_OBJ=$ACC_OBJ MAXC=$MAXC -> $OUT"

$PY iss_analysis/panel_design.py \
    --cache_dir $CACHE --out_dir $OUT \
    --level $LEVEL --accuracy_objective $ACC_OBJ --lam $LAM --efficiency $EFF --n_select $NSEL \
    --min_max_subclass_mean $MIN_EXPR --min_tau $MIN_TAU --max_candidates $MAXC

echo "DONE: $OUT"
