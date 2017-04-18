#!/bin/sh
#
#SBATCH -o model_curves/base.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=ass502@nyu.edu # send-to address
#SBATCH -t 0-5:00 

module load pytorch/intel

module load torchvision

python main.py --niter=5
