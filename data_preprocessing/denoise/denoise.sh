#!/bin/sh
 
#SBATCH --job-name=test
#SBATCH --output=test-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=test-%A.err  # Standard error of the script
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=30G  # Memory in GB (Don't use more than 126G per GPU) BEFORE: 36G
#SBATCH -w unimatrix2

# activate corresponding environment
# source ../../../../.venv-p311/bin/activate
source ../../../../.venv-p311-cu116/bin/activate


pwd

echo "test: nvidia-smi"
nvidia-smi 
echo "test: torch version"
python -c "import torch; print(torch.__version__)"


datasets=(
    # "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/THYROID/val/val18"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/test_heart/val/val0"
)

for dataset in "${datasets[@]}"; do
    echo "Preprocessing for dataset: $dataset"

	python demo.py \
	--task Deraining \
	--input_dir "$dataset"/images \
	--result_dir "$dataset"/mpr_derained

done


