#!/bin/sh
#
#SBATCH -o cifar/conditional_40/conditional_40_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-45:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main_with_conditional.py --niter=20 --dataset='cifar10' --outf='cifar/conditional_40/' --netG='cifar/conditional/netG_epoch_19.m' --netD='cifar/conditional/netD_epoch_19.m'
