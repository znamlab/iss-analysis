# BARseq2 gene-panel design — handover

## Goal
Select 200–400 genes for BARseq2 cell-type classification of cortical neurons, ranked by
"usefulness", capturing **discrete cell types and continuous gradients**, targeting neuronal
**subclass** (Pvalb, L5 IT, …) and rewarding **cluster** resolution. Reference = Allen
**scRNA-seq**; broadly useful across neocortex, emphasis on V1 (VISp).

## Environment / how to run
- The `iss_analysis` package **does not import** in the env (`__init__` pulls in
  `iss_preprocess` → missing `brainglobe_atlasapi`). All panel code is **self-contained**
  and loaded as top-level modules: `sys.path.insert(0, ".../iss_analysis"); import panel_design`.
- Interpreter: `/camp/home/znamenp/.conda/envs/iss-analysis/bin/python` (py3.10; numpy,
  scipy, pandas, sklearn, h5py, scanpy, leidenalg, igraph; **no** `Anaconda3/2020.07` module).
- SLURM partition is **`ncpu`** (not `hmem`); jobs need ≈64 GB, a few hours.

## Code (in `iss_analysis/`, all committed)
- **`panel_design.py`** — selection pipeline + CLI (`python iss_analysis/panel_design.py …`).
  Key funcs: `load_allen2020_streaming` (Stage 0 streaming loader → cache),
  `normalize_library`/`normalize_bundle` (per-cell depth normalisation),
  `filter_candidate_genes` (Stage 1), `marginal_usefulness` (Stage 2 MI),
  `build_neighbor_graph` (Stage 3), `nb_coeffs`/`_cell_loglik` (scalable NB classifier),
  `greedy_accuracy`(hierarchical)/`greedy_accuracy_single`(one level) with
  `objective∈{soft_acc,logpost,accuracy}`, `greedy_overlap` (geneBasis-style),
  `reciprocal_rank_fusion`, `_prepare_selection` (shared Stages 0–3 setup),
  `run_selection` (parallel accuracy+overlap fuse), `run_selection_stepwise`
  (sequential seed-chained stages), `evaluate_curve_pure` (per-region accuracy curve), `main`.
  Two cross-cutting **options** (both via `_prepare_selection`/`run_selection_stepwise`):
  - **custom seed genes** (`seed_genes=…`, CLI `--seed_genes_file`/`--seed_genes_col`/`--seed_genes`)
    — force-include a starting gene set (kept through filter + MI cap; occupy stage-1's first
    slots) instead of just the conventional markers.
  - **region-specific stage optimisation** — a step-wise stage `("subclass",n,"MOs_FRP")`
    (CLI `--stages subclass:n:MOs_FRP`) restricts that accuracy stage's *evaluation cells* to a
    cortical region (global NB means kept); `evaluate_curve_pure(region_masks=…)` reports
    per-region accuracy (VISp + MOs) into `accuracy_curve.csv`.
- **`panel_eval.py`** — `load_allen2020`/`load_allen2018` (→ standard ds dict, **per-cell
  normalised on load**), `make_bundle` (run selection on any dataset), `evaluate_panel`,
  `reference_panels`, `group_means`. Datasets are depth-matched so eff=0.1 applies to both.
- **`panel_plots.py`** — reusable plots/eval: `classify`, `embed`, `selection_curves`
  (now plots whole-cortex + VISp + **MOs** accuracy if those columns exist; tolerates a missing
  `selection_meta.npz` for manually-ordered panels), `expression_across_subclasses`,
  `per_gene_boxplots` (full-depth **normalised** single-cell counts; `eff=None` → no
  down-sampling), `confusion_grid` (any `region=…`), `umap_grid`, `leiden_vs_labels`,
  `panel_table`, `region_accuracy`, `expression_budget_curve` (**median per-cell summed
  expression** — see findings). `results/make_all_figures.py` now also emits
  `confusion_mos_{subclass,cluster}.png` and skips the selection-curve figure for manual panels.

## Pipeline (Stages 0–5)
0. Stream the dense Allen-2020 HDF5 (genes×cells) in chunk-aligned blocks → cache
   (`results/panel_cache`): cluster/subclass means, gene stats, 120k-cell subsample.
1. Biophysical filter: peak-subclass mean ≥ `min_max_subclass_mean`, τ ≥ 0.3, drop
   artifacts; **conventional markers force-kept** (Gad1/Slc17a7/Pvalb/Sst/Vip/… in
   `CONVENTIONAL_MARKERS`).
2. Per-gene marginal mutual information under BARseq down-sampling → ranking + pool cap
   (`max_candidates`, top by MI).
3. Reference manifold (HVG→PCA→kNN) for the continuous objective.
4. Two greedy selections (lazy CELF): (a) classification accuracy at the chosen `level`,
   (b) kNN-graph overlap; fused by reciprocal-rank fusion.
5. Evaluation: accuracy vs panel size (subclass+cluster; whole cortex + VISp + MOs), curves,
   confusion (incl. per-region), UMAP. (`run_selection_stepwise` is the alternative to the
   Stage-4 fuse — sequential seed-chained stages instead of two parallel greedies + fusion.)

## How to run selections / analyses
- One panel: `sbatch --export=ALL,LEVEL=subclass,MIN_EXPR=0.85 select_panels.sh`
  (LEVEL∈{subclass,cluster,hierarchical}; ACC_OBJ∈{soft_acc,logpost,accuracy};
  output dir `results/panel_{LEVEL}_e{MIN_EXPR}_t{MIN_TAU}[_soft_acc]`).
- **Step-wise panel** (`select_panels.sh` `STEPWISE=1` branch; output
  `results/panel_{PREFIX}stepwise_e{MIN_EXPR}_t{MIN_TAU}`): set vars via `export` then
  `sbatch --export=ALL …` (comma-containing `STAGES`/`SEED_GENES` can't go in `--export=KEY=VAL`).
  - `STAGES="subclass:200,cluster:100,overlap:100"` — `obj:n_add[:region]`, cumulative sizes;
    obj∈{subclass,cluster,overlap}; optional region restricts an accuracy stage to e.g. `MOs_FRP`.
  - `SEED_FILE=<csv> SEED_COL=<gene-col idx>` (codebook GII,barcode,gene → 2; `probes_v3.csv` → 1)
    and/or `SEED_GENES="Gad1,Slc17a7,Vip,Sst"` — forced starting genes (override conventional seed).
  - e.g. `export STEPWISE=1 PREFIX=pv3mos_ MIN_EXPR=3 MIN_TAU=0.3 STAGES="subclass:200:MOs_FRP,cluster:100:MOs_FRP" SEED_FILE=probes_v3.csv SEED_COL=1; sbatch --export=ALL select_panels.sh`
- Figures for a run: `sbatch --export=ALL,PANEL_DIR=results/panel_subclass_e0.85 analyse_panels.sh`
  (runs `results/make_all_figures.py`; works for manual panels too — skips the selection curve).
- Cross-run comparisons: `results/compare_panels_curves.py` (accuracy + budget vs #genes
  across level×min_expr×{hard,soft_acc}); `results/compare_objectives.py` (hard vs soft);
  `results/random_baseline.py <dir> <min_expr>` (selected vs random-from-filtered-pool).

## Data
- Allen 2020 (Yao 2021): `/camp/home/znamenp/home/shared/resources/allen2020/` —
  `expression_matrix.hdf5` stores **dense** `data/counts` 31053 genes × 1,169,320 cells
  (10x UMIs). Align metadata by `reindex(samples)`. Neocortical neurons ≈930k, 16 clean
  subclasses / 215 clusters (allocortical ENT/PPP/TPE/SUB subclasses dropped). VISp ≈30k.
- Allen 2018 (Tasic VISp, SMART-seq): `/camp/.../allen2018/` — independent validation set,
  cached (depth-matched) at `results/panel_cache_2018`. ~13.5k neurons, 16 subclasses/94 clusters.

## Key findings / decisions
- **Per-cell library-size normalisation matters and is now ON by default**
  (`normalize=True`). The original NB path used raw counts; depth (CV 0.48) is confounded
  with cell type (L5 PT ~18k vs Vip ~6k UMIs). Normalising → **+9 pt cluster, +15 pt VISp
  cluster** at 400 genes. NB classification reduces to a matrix product `X·Aᵀ+offset`.
- **`soft_acc` objective dominates hard accuracy** (higher subclass+cluster acc *and* higher
  expression budget); default `accuracy_objective="soft_acc"`.
- **Efficiency=0.1** is the default (0.01 is unusable — the real rolony panel scores below
  baseline there; rolony→0.79 subclass at 0.1).
- **min_expr ↔ candidates (normalised data, τ≥0.3):** 3→680, 1.5→1291, 1.0→1796,
  **0.85→~2000**, 0.8→2116.
- **Cross-dataset (2×2):** home-field advantage is real & symmetric; the **Allen-2020 panel
  generalises better** to Tasic than the Tasic panel does to 2020 → select on broad 2020.
  Low-expression genes transfer less robustly across platforms.
- **vs published Chen-lab panels** (rolony/BARseq2 76, codebook_88 83, csm_and_v1 ~429):
  our panels win clearly on broad Allen-2020 (e.g. +10 pt subclass at matched size) and are
  competitive on VISp/Tasic; ours uses higher-expression genes. Codebooks live in
  `iss-preprocess/iss_preprocess/call/`. (csm_and_v1 = cell-surface-molecule panel, ~429 genes.)
- **Random baseline:** selection beats random-from-the-same-filtered-pool by ~+23/+25 pts
  (subclass/cluster) at 400 genes — the greedy, not just the filter, does the work.
- **Expression budget** = cumulative **median per-cell summed expression** over the panel
  (`expression_budget_curve`: for each panel size, sum the genes' counts within each cell then take
  the MEDIAN across cells). Changed from mean→median (the per-cell summed-expression distribution
  is right-skewed; optical crowding is set by the *typical* cell). Uses NORMALISED counts (depth-
  equalised to the median library); full depth (×efficiency for expected rolonies). The change
  matters most for marker-heavy sets — e.g. probes_v3 median 81 vs old mean 108.7 (high-expression
  genes inflate the mean). Reference panels are ordered by descending per-gene mean (steepest curve).

## Current runs (`results/`)
- **12-panel batch:** `panel_{subclass,cluster}_e{0.85,1.5,3}[_soft_acc]` (level × min_expr ×
  {hard, soft_acc}); each has `gene_ranking.csv`, `accuracy_curve.csv`, `selection_meta.npz`
  (and figures if analysed).
- `panel_2018` (Tasic-selected), `panel_norm`/`panel_cluster_norm`/`panel_minexpr1`
  (earlier exploratory, normalised/min-expr variants — superseded by the batch).
- **Step-wise panels** (all `soft_acc`, e3, τ0.3): `panel_stepwise_e3_t0.3` (conventional seed,
  200/100/100 subclass→cluster→manifold), τ-sweep `panel_stepwise_e3_t{0.4,0.5}`, and
  `panel_stepwise_e3_t0.3_eff0.3` (selected at eff=0.3). Comparison driver
  `results/plot_stepwise_compare.py` + L2/3 `results/compare_l23_stepwise.py`.
- **Seeded panels** (codebook seed + step-wise): `panel_cb83_stepwise_e3_t0.3` (+`cb83small`),
  `panel_cb88_stepwise_e3_t0.3` (+`cb88small`); generalised comparison
  `results/plot_stepwise_seed_compare.py <tag> <label::dir::b1,b2 …>`.
- **probes_v3 family** (seed = the 104 genes in `probes_v3.csv`, gene col idx 1): `panel_probes_v3`
  (manual order = first appearance), `panel_pv3sc_stepwise_e3_t0.3` (subclass:200→cluster:100),
  `panel_pv3sc300_…` (300→100), `panel_pv3seed_…` (200 subclass → +50 subclass@MOs → +100 cluster),
  `panel_pv3mos_…` (subclass:200→cluster:100, **both stages @MOs_FRP**). Cortex-vs-MOs accuracy
  overlay: `results/pv3_cortex_vs_mos_accuracy.png` (rows = cortex/VISp/MOs, cols = subclass/cluster).
- Comparison outputs: `results/panel_curves_comparison.{png,csv}`, plus per-dir figure sets.
- Notebook: `notebooks/cell_types.ipynb` (clean driver over the modules; set `PANEL_DIR`).

## L2/3 transcriptomic-gradient reconstruction (VISp / V1)
Evaluates how well each stored panel reconstructs the **continuous** L2/3 IT gradient in V1
(Xie et al. 2025 *PNAS* 122(7):e2421022122 / Cheng et al. 2022). Ground truth = the **286 L2/3
cell-type-identity genes** (Dataset S1, parsed to `results/l23_reference/identity_genes_286.csv`;
282/286 in the Allen vocab) on the **3,426 VISp `L2/3 IT CTX`** Allen-2020 neurons. Those cells
sit in 4 contiguous HDF5 column-chunks → pulled directly from `expression_matrix.hdf5` in ~50 s
and cached at `results/l23_reference/visp_l23_cells.npz` (the 120k subsample has only 424 such
cells — too few). PNAS supp came via the Europe-PMC zip endpoint (PNAS direct dl is Cloudflared).
- **Code:** `iss_analysis/l23_gradient.py` (standalone module — `load_reference_cells`,
  `load_identity_genes`, `random_pool`, `embed_pca`, `build_reference`/`get_reference`
  [caches `reference_gradient.npz`; one-time ~100 s scanpy/numba JIT for the diffusion coord],
  `score_panel`/`score_panel_reps`). Driver `results/compare_l23_gradient.py` →
  `results/l23_gradient_comparison.{png,csv}` (~11 min, reference cached). PC-plane figure
  `results/plot_l23_pc_embeddings.py` → `results/l23_gradient_pc_embeddings.png`.
- **Method:** reference manifold = identity-gene PCA (PC1 = A→B→C continuum; PC1–PC2 = triangle)
  + root-free diffusion component DC1 (|corr(PC1,DC1)|=0.92) + k=15 ref graph. Panels are
  BARseq-down-sampled at eff=0.1 then PCA'd (mirrors `panel_plots.embed` preproc). 6 metrics:
  **PC1/DC1 coordinate recovery** (CV-ridge Spearman ρ / R² — headline; numpy ridge, the env's
  sklearn `Ridge` hits a scipy `sym_pos` removal), 2D triangle R², kNN-overlap (reuses
  `_mean_knn_overlap`), trustworthiness/continuity, pairwise-distance Spearman. (kNN-overlap is
  near-floor in absolute terms — a down-sampled continuum has unstable exact neighbours — but
  still ranks panels; coordinate recovery is the robust headline.)
- **Findings (coord ρ @400 genes, eff=0.1):** all 6 metrics agree. **soft_acc > hard** and
  **cluster-level > subclass-level** for gradient reconstruction. Best = `cluster_e3_soft_acc`
  **0.867 ≈ identity-gene ceiling 0.869** — a general 400-gene panel reconstructs the V1 L2/3
  continuum almost as well as the 282 purpose-built genes. Worst stored = `subclass_e0.85` 0.784;
  random-from-filtered-pool floor 0.63. min_expr effect small (0.85/1.5/3 within ~0.01 among
  soft_acc). Ranking stable across down-sampling seeds (rank-Spearman 0.95, top-3 identical).

## Step-wise selection, seeding & region optimisation (findings)
- **Step-wise (seed-chained) selection** builds ONE cumulative panel: stage 1 maximises subclass
  accuracy, stage 2 adds genes that most improve cluster accuracy *given* stage 1, stage 3 adds
  manifold (kNN-overlap) genes *given* stages 1–2 (`run_selection_stepwise`, default
  200/100/100). Result = **best-of-each**: matches the cluster-level panel on cluster acc *and*
  the subclass-level panel on subclass acc *and* has the best manifold preservation.
- **τ sweep:** τ=0.3 is best; τ=0.4/0.5 progressively worse, and **τ=0.5 can't fill 400 genes**
  (only ~339 candidates clear `min_expr=3` after normalisation). **eff=0.3-selected ≈ eff=0.1**
  on classification (selecting under a more optimistic efficiency barely changes the panel).
- **Seeding with an existing codebook is ~free.** Forcing codebook_83 / codebook_88 (+Gad1/
  Slc17a7/Sst/Vip; codebook_88 has none of the standard E/I markers) as the stage-1 seed matches
  the conventional-seed panel on every classification metric and is marginally *better* on the
  L2/3 gradient (cb88-400 PC1 ρ 0.869 ≈ identity ceiling). ⚠️ a custom seed *replaces* the
  conventional-marker forcing — codebook_88 dropped Gad2/Slc32a1/Slc17a6/Satb2 (re-add via
  `SEED_GENES` if you want them guaranteed).
- **Region-specific optimisation works.** A `subclass:50:MOs_FRP` stage raises **MOs** subclass
  accuracy ~+2.2 pt vs the whole-cortex panel at matched size, region-specifically (VISp/cortex
  unchanged). MOs-only optimisation (`pv3mos`) gives the best MOs **cluster** accuracy (0.667 vs
  0.639/0.594) at only ~0.3–1 pt cost to whole-cortex/VISp. Caveat: the MOs-subclass edge peaks
  right after the MOs stage and erodes as later whole-cortex cluster genes are added.
- **MOs label = `MOs_FRP`** in `sub_region` (4,405 cells). VISp = `VISp` (3,714).

## Open items / next steps
- **Pick the final panel.** The **step-wise e3/τ0.3 soft_acc** panel is the current
  recommendation (best-of-each on subclass/cluster/manifold); seed it with the chosen codebook
  (free) and add a region stage if a region (e.g. MOs) is a priority. Decide total size and the
  stage split (200/100/100 vs more subclass vs region-targeted).
- **Uncommitted code changes** (working tree, not yet committed): custom seed genes +
  region-specific stage optimisation in `panel_design.py`; MOs/per-region accuracy curve +
  median expression budget + manual-panel robustness in `panel_plots.py`; `STEPWISE`/`SEED_*`
  branch in `select_panels.sh`. (`results/` is gitignored.)
- Re-confirm BARseq2 `--efficiency` against the real platform sensitivity.
- Optionally split the probe-ordering panel into nested oligo pools (deferred design discussion).
