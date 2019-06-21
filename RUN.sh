#!/bin/bash

#SBATCH --mem 24000M
#SBATCH --gres=gpu:1
#SBATCH --output="./log.slurm"
#SBATCH --time 72:00:00
#SBATCH -C 'rhel7'

echo $USER
if [ $USER == "tarch" ]; then
  email="taylor.archibald@byu.edu"
else
  email="masonfp@byu.edu"
fi

echo "$email"

#SBATCH --mail-user=masonfp@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge

activate "/fslhome/$USER/fsl_groups/fslg_hwr/compute/env/deepwriting/"
#export PATH="/fslhome/$USER/fsl_groups/fslg_hwr/compute/env/hwr_env/bin:$PATH"
which python

python create_samples.py -S ./experiment -M ./tf-1514981744-deepwriting_synthesis_model/ -QL


