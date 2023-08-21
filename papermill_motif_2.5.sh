#!/bin/bash
#PBS -N Papermill_Motif_2.5_distributed
#PBS -P fx09
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=1:00:00 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -l wd
#PBS -J 0-12:2
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

tf_array=( Ase Dpax2 Fer1 gtSuH Hairless Insv Scute Sens Seq Ttk69 wtElba1 wtElba2 wtElba3 wtInsv)

output_notebook_path=TF_Motif_Notebook/${tf_array[$PBS_ARRAY_INDEX]}_2.5.output.ipynb
output_notebook_path_1=TF_Motif_Notebook/${tf_array[$PBS_ARRAY_INDEX+1]}_2.5.output.ipynb

papermill TF_motif_2.5.ipynb $output_notebook_path -p the_tf ${tf_array[$PBS_ARRAY_INDEX]}

papermill TF_motif_2.5.ipynb $output_notebook_path_1 -p the_tf ${tf_array[$PBS_ARRAY_INDEX + 1]}






