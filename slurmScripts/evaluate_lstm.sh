#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=00:10:00 
#SBATCH --error=../logs/evaluate_weights_lstm.err 
#SBATCH --output=../logs/evaluate_weights_lstm.out
#SBATCH --job-name=evaluate_weights_lstm
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to GenrePrediction directory.
cd ..
luajit evaluateModel.lua 

echo "\nFinished with exit code $? at: `date`"

