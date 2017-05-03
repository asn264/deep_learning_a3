#!/bin/sh
#
#SBATCH -o cifar/conditional_LARGE_CLASS_EMBEDDING/conditional_40_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-90:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main_with_conditional_LARGE_CLASS_EMBEDDING.py --niter=40 --dataset='cifar10' --outf='cifar/conditional_LARGE_CLASS_EMBEDDING/'
