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
# model_array=(trans_unet unet se_unet)
model_array=(trans_unet)

peak_calling_method="_2.5.fimo_labels"
for i in "${model_array[@]}"
do
    horovodrun -np 4 python3 "$path"/deep_tf.py -m $i -f $path -p $peak_calling_method -t Ase Dpax2 Fer1 gtSuH Hairless Insv Scute Sens Seq Ttk69 wtElba1 wtElba2 wtElba3 wtInsv
done

python multimodel_performance_summary.py



