#!/bin/bash

#SBATCH --mem 200M
#SBATCH --time 01:00:00

echo $USER
if [ $USER == "tarch" ]; then
  email="taylor.archibald@byu.edu"
else
  email="masonfp@byu.edu"
fi

echo "$email"

#SBATCH --mail-user= masonfp@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

echo "did it email you?"