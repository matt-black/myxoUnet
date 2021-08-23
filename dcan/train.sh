#!/usr/bin/bash
#SBATCH --job-name=dcan-train
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

python train.py --data="../supp/dcankc" \
       --crop-size=263 --data-statnorm \
       --learning-rate=.0001 --batch-size=1 --epochs=10 \
       --dcan-depth=5 --dcan-wf=4 --dcan-kernel-size=3 \
       --dcan-batchnorm --dcan-upmode="upsample" --dcan-outdim=256 \
       --print-freq=1 --save
