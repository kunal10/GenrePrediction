#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=02:00:00 
#SBATCH --error=../logs/train_weights_blstm.err 
#SBATCH --output=../logs/train_weights_blstm.out
#SBATCH --job-name=train_weights_blstm
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to GenrePrediction directory.
cd ..
# Train BLSTM
luajit blstm.lua 

echo "\nFinished with exit code $? at: `date`"

