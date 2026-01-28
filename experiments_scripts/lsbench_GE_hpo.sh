#!/bin/bash
#SBATCH --job-name=scenario_lsbench_GE_hpo
#SBATCH --output=scenario_lsbench_GE_hpo_%a.txt
#SBATCH --mem-per-cpu=2541
#SBATCH --cpus-per-task=1
#SBATCH --array=0-0
#SBATCH --time=01:00:00
#SBATCH --account=rwth1938

# List of datasets
datasets=('add4' 'mul3' 'alu4' 'count4' 'dec4' 'enc8' 'epar8' 'mcomp4' 'icomp5')

# Calculate dataset and seed
N_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10))
# Get the dataset name
dataset=${datasets[$dataset_idx]}

# Run the Python script
python3 -m examples.experiments.scenario_lsbench --algo GE -o -s $seed -d $dataset
