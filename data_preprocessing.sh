#!/bin/bash
#PBS -N Data_preprocessing
#PBS -P fx09
#PBS -r y
#PBS -q normal  
#PBS -l storage=gdata/ik06 
#PBS -l walltime=12:00:00 
#PBS -l ncpus=12
#PBS -l mem=190GB
#PBS -l wd
#PBS -M ke.ding@anu.edu.au
#PBS -m e

###########################################
# This script is for data preprocessing
###########################################

# setup conda environment 
# change the path to your own conda directory
source /g/data/ik06/stark/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate deep_tf

path="/g/data/ik06/stark/scTFNet/"

# by default the flanking region length is 1000
flanking_region_length=1000

###########################################
## for 2.5 time point
###########################################

peak_calling_method="_2.5.fimo_labels"

array=( Ase Dpax2 Fer1 gtSuH Hairless Insv Scute Sens Seq Ttk69 wtElba1 wtElba2 wtElba3 wtInsv )
for i in "${array[@]}"
do
	python3 "$path"data_preprocessing.py -tf $i -p $path -m $peak_calling_method -f $flanking_region_length
done

###########################################
## for 2.5 multi-modal
###########################################

python3 "$path"multi_data_preprocessing.py -p $path -f $flanking_region_length -m $peak_calling_method -tf Ase Dpax2 Fer1 gtSuH Hairless Insv Scute Sens Seq Ttk69 wtElba1 wtElba2 wtElba3 wtInsv

###########################################
## for 6.5 time point
###########################################

peak_calling_method="_6.5.fimo_labels"

array=( Ase Chn Fer1 gtSuH Ham Insv Scrt Sens Ttk )
for i in "${array[@]}"
do
	python3 "$path"data_preprocessing.py -tf $i -p $path -f $flanking_region_length -m $peak_calling_method
done

###########################################
## for 6.5 multi-modal
###########################################

python3 "$path"multi_data_preprocessing.py -p $path -f $flanking_region_length -m $peak_calling_method -tf Ase Chn Fer1 gtSuH Ham Insv Scrt Sens Ttk