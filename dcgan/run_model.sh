#!/bin/sh
#
#SBATCH -o cifar/baseline_normalized/baseline_20_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-40:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main.py --niter=20 --dataset='cifar10' --outf='cifar/baseline_normalized/'
