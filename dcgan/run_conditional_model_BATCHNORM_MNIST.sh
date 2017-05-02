#!/bin/sh
#
#SBATCH -o mnist/conditional_BATCHNORM/conditional_BATCHNORM.out        # STDOUT
#SBATCH --mail-type=ALL      # notifications for job done & fail
#SBATCH --mail-user=asn264@nyu.edu # send-to address
#SBATCH -t 0-55:00 
#SBATCH --mem=10GB

module load pytorch/intel
module load torchvision

python main_with_conditional_BATCHNORM_MNIST.py --niter=20 --dataset='mnist' --outf='mnist/conditional_BATCHNORM/'

