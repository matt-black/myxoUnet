#!/usr/bin/bash
#SBATCH --job-name=tl-proc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=mb46@princeton.edu

module purge
module load anaconda3/2020.11
conda activate pyt

python process_full_movie.py --checkpoint="/scratch/gpfs/mb46/2021-09-21_127" \
       --data="/scratch/gpfs/mb46/frzE_highP/view1_2" --data-format="ktl" \
       --training-data="/home/mb46/myxoUnet/supp/prob_ds" \
       --output="/scratch/gpfs/mb46/frzE_highP/view1_2/bunet_127"