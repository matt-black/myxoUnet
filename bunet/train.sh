#!/usr/bin/bash
#SBATCH --job-name=unet-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mail-type=end
#SBATCH --mail-user=mb46@princeton.edu

module purge
module load anaconda3/2020.11
conda activate pyt

python train.py --data="../supp/test" --data-global-stats \
       --crop-size=256 --batch-size=2 --epochs=100 \
       --step-lr="100,0.1" --learning-rate=.0001\
       --unet-depth=3 --unet-wf=6 \
       --unet-upmode="upconv" --unet-downmode="maxpool"
       --print-freq=1 --save
