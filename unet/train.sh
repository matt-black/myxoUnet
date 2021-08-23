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

python train.py --data="../supp/testkc" --loss="wce" --num-classes=3 \
       --crop-size=256 --data-nplicates=1 --data-statnorm \
       --learning-rate=.0001 --batch-size=1 --epochs=200 \
       --unet-depth=3 --unet-wf=6 --unet-batchnorm --unet-upmode="upconv" \
       --print-freq=1 --save
