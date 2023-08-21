#!/bin/bash
#PBS -N Papermill_Motif_6.5_distributed
#PBS -P fx09
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=1:00:00 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -l wd
#PBS -J 0-8:1
#PBS -j oe
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

tf_array=(Ase Chn Fer1 gtSuH Ham Insv Scrt Sens Ttk )

output_notebook_path=TF_Motif_Notebook/${tf_array[$PBS_ARRAY_INDEX]}_6.5.output.ipynb

papermill TF_motif_6.5.ipynb $output_notebook_path -p the_tf ${tf_array[$PBS_ARRAY_INDEX]}







