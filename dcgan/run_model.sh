#!/bin/sh
#
#SBATCH -o cifar/baseline/baseline_10_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-30:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main.py --niter=10 --dataset='cifar10' --outf='cifar/baseline/'
