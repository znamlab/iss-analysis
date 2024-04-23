#!/bin/bash
#SBATCH --job-name=pick-genes
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=384G
#SBATCH --partition=hmem
#SBATCH --mail-type=END,FAIL

ml Anaconda3/2020.07

source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate iss-analysis

echo classifying by ${CLASSIFY}
echo selecting genes with efficiency = ${EFFICIENCY} and subsample = ${SUBSAMPLE}

cd /camp/home/znamenp/home/users/znamenp/code/iss-analysis
pick_genes --efficiency ${EFFICIENCY} --subsample ${SUBSAMPLE} --classify ${CLASSIFY} results/
