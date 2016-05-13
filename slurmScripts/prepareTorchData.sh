#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 10
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=00:15:00 
#SBATCH --error=../logs/prepareTorchData.err 
#SBATCH --output=../logs/prepareTorchData.out
#SBATCH --job-name=prepareTorchData
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# Required for npy4th
module load gcc/4.9.1
# cd to GenrePrediction directory.
cd ..
# Convert numpy features to torch format
luajit prepareTorchData.lua

echo "\nFinished with exit code $? at: `date`"

