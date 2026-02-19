#!/bin/bash
#SBATCH --job-name=scenario_lsbench_CGP_hpo
#SBATCH --output=scenario_lsbench_CGP_hpo_%a.txt
#SBATCH --mem-per-cpu=6000
#SBATCH --cpus-per-task=1
#SBATCH --array=1-90
#SBATCH --time=7-00:00:00
#SBATCH --qos=long

# if you installed Miniforge to a different location, change the path accordingly
#export CONDA_ROOT=$HOME/miniconda3
#source $CONDA_ROOT/etc/profile.d/conda.sh
#export PATH="$CONDA_ROOT/bin:$PATH"
#conda activate tinyverse

# List of datasets
datasets=('add4' 'mul3' 'alu4' 'count4' 'dec4' 'enc8' 'epar8' 'mcomp4' 'icomp5')

# Calculate dataset and seed
N_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID / 9))
seed=$((SLURM_ARRAY_TASK_ID % 9))
# Get the dataset name
dataset=${datasets[$dataset_idx]}

# Run the Python script
python3 -m examples.experiments.scenario_lsbench  --algo CGP -o -s $seed -d $dataset