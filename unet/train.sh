#!/usr/bin/bash
#SBATCH --job-name=unet-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mail-type=end
#SBATCH --mail-user=kc32@princeton.edu

module purge
module load anaconda3/2020.11
conda activate cells

python train.py --data="/home/kc32/trainingdata/20210917border" --data-global-stats \
       --loss="dsc" --num-classes=5 --crop-size=260 \
       --learning-rate=.0001 --batch-size=2 --epochs=500 \
       --reduce-lr-plateau="0.1,10,0.0001", \
       --unet-depth=5 --unet-wf=6 --unet-batchnorm --unet-upmode="upconv" \
       --print-freq=1 --save 
