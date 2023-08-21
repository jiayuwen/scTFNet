#!/bin/bash
#PBS -N Distributed_DeepTF
#PBS -P fx09
#PBS -r y
#PBS -q gpuvolta  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=3:00:00 
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=300GB
#PBS -J 0-8:1
#PBS -j oe
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

path="/g/data/ik06/stark/scTFNet/"
output_path="/g/data/ik06/stark/scTFNet/output_new"
peak_calling_method="_6.5.fimo_labels"

tf_array=(Ase Chn Fer1 gtSuH Ham Insv Scrt Sens Ttk )
# model_array=(trans_unet unet se_unet)
model_array=(trans_unet)

for i in "${model_array[@]}"
do
    horovodrun -np 4 python3 "$path"deep_tf.py -m $i -f $path -p $peak_calling_method -t ${tf_array[$PBS_ARRAY_INDEX]}
done




