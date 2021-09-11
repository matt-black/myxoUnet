#!/usr/bin/bash
#SBATCH --job-name=mrcnn-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=mb46@princeton.edu

module purge
module load anaconda3/2020.11
conda activate pyt

python train.py --data="/scratch/gpfs/mb46/mrcnn_ds" --epochs=100 \
       --crop-size=256 --hidden-layer=256 --box-detections-per-img=200 \
       --save
