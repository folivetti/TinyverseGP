#!/bin/bash
#SBATCH --job-name=scenario_srbench_CGP_hpo
#SBATCH --output=scenario_srbench_CGP_hpo_%a.txt
#SBATCH --mem-per-cpu=6000
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99

# List of datasets
datasets=('522_pm10' '678_visualizing_environmental' '192_vineyard' '1028_SWD'
'1199_BNG_echoMonths' '210_cloud' '1089_USCrime' '1193_BNG_lowbwt'
'557_analcatdata_apnea1' '650_fri_c0_500_50')

# Calculate dataset and seed
N_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10))
# Get the dataset name
dataset=${datasets[$dataset_idx]}

# Run the Python script
python3 -m examples.symbolic_regression.scenario_srbench --algo CGP -o -s $seed -d $dataset