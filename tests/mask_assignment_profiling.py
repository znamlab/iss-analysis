"""Script to profile the mask assignment function."""
# %%
import numpy as np

# generate some data
rng = np.random.default_rng(42)
nbarcodes = 100
nmasks = 5000
mask_positions = rng.uniform(0, nmasks, (nmasks, 2))
print(f"Generating data for {nbarcodes} barcodes", flush=True)
all_positions = []
barcode_id = []
for bc_id in range(nbarcodes):
    ncells = int(rng.uniform(1, 20))
    assignment = rng.choice(np.arange(nmasks), ncells)
    spotpercell = rng.exponential(10, ncells).astype(int) + 5
    assigned_spots = np.sum(spotpercell)
    background_spots = int(np.sum(spotpercell) * 0.3)
    print(
        f"ncells: {ncells}, nmasks: {nmasks}, assigned spots: {assigned_spots}, background_spots: {background_spots}"
    )

    # Generate random cell positions

    spot_positions = []
    for assi, nspots in zip(assignment, spotpercell):
        cell_pos = mask_positions[assi]
        sp_pos = rng.multivariate_normal(cell_pos, np.eye(2) * 10, nspots)
        spot_positions.extend(sp_pos)
    spot_positions.extend(rng.uniform(0, nmasks, (background_spots, 2)))
    spot_positions = np.c_[spot_positions]
    all_positions.extend(spot_positions)
    barcode_id.extend([bc_id] * spot_positions.shape[0])
print(f"Data generated", flush=True)

# %%
# run the single assignment function
if False:
    from iss_analysis.barcodes.probabilistic_assignment import assign_single_barcode

    print("Calling assign_single_barcode", flush=True)
    assign_single_barcode(
        spot_positions=spot_positions,
        mask_positions=mask_positions,
        max_distance_to_mask=600,
        inter_spot_distance_threshold=50,
        background_spot_prior=0.0001,
        p=0.8,
        m=0.08,
        spot_distribution_sigma=100,
        max_spot_group_size=10,
        max_total_combinations=10000,
        verbose=0,
        mask_assignments=None,
        max_iterations=100,
        debug=False,
    )

# %%
# run the full assignment function
import pandas as pd
from iss_analysis.barcodes.probabilistic_assignment import assign_barcodes_to_masks

all_positions = np.c_[all_positions]
spot_df = pd.DataFrame(index=np.arange(len(all_positions)), columns=["x", "y", "bases"])
spot_df["x"] = all_positions[:, 0]
spot_df["y"] = all_positions[:, 1]
spot_df["bases"] = barcode_id
masks = pd.DataFrame(index=np.arange(len(mask_positions)), columns=["x", "y"])
masks["x"] = mask_positions[:, 0]
masks["y"] = mask_positions[:, 1]
assign_barcodes_to_masks(
    spots=spot_df,
    masks=masks,
    p=0.8,
    m=0.08,
    background_spot_prior=0.0001,
    spot_distribution_sigma=50,
    max_iterations=100,
    max_distance_to_mask=500,
    inter_spot_distance_threshold=50,
    max_spot_group_size=5,
    max_total_combinations=1e6,
    verbose=1,
    base_column="bases",
    n_workers=10,
    run_by_groupsize=False,
)

# %%
