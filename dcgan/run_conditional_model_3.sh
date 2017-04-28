#!/bin/sh
#
#SBATCH -o cifar/conditional_3/conditional_20_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-75:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main_with_conditional_3.py --niter=20 --dataset='cifar10' --outf='cifar/conditional_3/'
