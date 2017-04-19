#!/bin/sh
#
#SBATCH -o model_curves/base_more_memory.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=ass502@nyu.edu # send-to address
#SBATCH -t 0-7:00 
#SBATCH --mem=10GB

module load pytorch/intel

module load torchvision

python main.py --niter=3
