#!/bin/bash
#PBS -N Keras_Tuner_Parameter_Search
#PBS -P fx09
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=24:00:00 
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=300GB
#PBS -l wd
#PBS -M ke.ding@anu.edu.au
#PBS -m be


###########################
#load modules for gpu support
module load cuda/11.4.1 
module load cudnn/8.2.2-cuda11.4 
module load nccl/2.10.3-cuda11.4 
module load openmpi/4.1.1


# setup conda environment 
# -- change the path to your own conda directory
source /g/data/ik06/stark/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate deep_tf

python3 keras_tuner_parameter_search_XL.py

