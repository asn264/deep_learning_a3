#!/bin/sh
#
#SBATCH -o mnist/conditional/conditional_20_epochs.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-45:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python mnist_main_with_conditional.py --niter=20 --dataset='mnist' --outf='mnist/conditional/'
