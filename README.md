# scTFNet

A deep learning model which can predict the transcription factors' binding sites in high resolution

<!-- TOC -->

- [Project Navigation](#project-navigation)
    - [Environment Installing](#environment-installing)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
    - [Ouput](#output)
    - [Notebook](#notebook)
    - [Parameter search](#parameter-search)
    - [Performance summary](#performance-summary)
    - [Plot](#motif-plot)


<!-- /TOC -->

## Environment Installing 
To inistall the environment using conda:

<code>conda env create -f deep_tf.yml</code>

## Data Preprocessing
To download the data (url will be updated soon):
<code>wget -P ./scTFNet/data/ [URL]</code>

To prepare the data on Gadi (current configuration: the preprocessed data will saved in tf.data.Dataset format):

<code>qsub data_preprocessing.sh</code>


## Model Training
To train the model for each transcription factor in parallel on Gadi (current configuration: 7~8 GPU Nodes and each node contains 4 GPUs):
    
<code>qsub distributed_training_*.sh</code>

To train the model to predict all transcription factors at the same time:

<code>qsub distributed_multi_model_*.sh</code>

    
## Output
under the */output* folder, we have
    
- /model  : saved model
- /plot   : training PR_AUC plot
- /result : evaluation metrics.csv
- /*_checkpoints : model checkpoints


## Notebook
- /proseq_genebody_expression-f*.ipynb
    
    *explore the idea of predicting PRO-seq genebody expression using DNA sequence and ATAC-seq with different flanking size configuration*

- /proseq_pausing_index-f*.ipynb
    
    *explore the idea of predicting PRO-seq pausing index using DNA sequence and ATAC-seq with different flanking size configuration

## Parameter Search
To do parameter search using HyperBand (keras tuner):

<code>qsub keras_tuner_parameter_search_XL.pbs</code>

To visualise the parameter search result in TensorBoard:

<code>tensorboard --logdir logs/keras_tuner_xl/tensorboard</code>

## Performance Summary

summarize the performance for each transcription factor from the */output* folder. The summary is saved in */performance_summary* folder

<code>python3 result_summarising.py -p $peak_calling_method </code>

summarize the performance for the joint model from the */output* folder. The summary is saved in */performance_summary* folder

<code>qsub multimodel_performance_summary.sh </code>

## Motif Plot

To generate saliency based TF motifs (the motif plot will be saved in */motif_logo_\*/*)

<code>qsub papermill_motif_*</code>
