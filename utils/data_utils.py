import pickle
# import scipy.io
import pyBigWig
import numpy as np
import pandas as pd
import os
import sys
import re
import tensorflow as tf


def generate_data_batch(if_train=True, if_test=False, the_tf=None, peak_method=None, flanking_region_len=1000, file_path='./'):
    path1 = file_path+'./data/dna_bigwig/'  # dna
    path2 = file_path+'./data/atac_bigwig/'  # atac
    # path4 = './data/hg38.phyloP100way.bw' # conservation score
    
    if peak_method != "_2.5.fimo_labels" :
        path3 = file_path+'./data/chipseq_6.5_bigwig/'  # 6.5 time point label
    else:
        path3 = file_path+'./data/chipseq_bigwig/' # 2.5 time point label
    
    if flanking_region_len == 1000:
        path4 = file_path+'./data/dm6_refGene_Prom_Flanking_1000.bed'
    else:
        path4 = file_path+'./data/dm6_refGene_Prom_Flanking_10k.bed'

    
    # open bigwig
    list_dna = ['A', 'C', 'G', 'T']
    dict_dna = {}
    for the_id in list_dna:
        dict_dna[the_id] = pyBigWig.open(path1 + the_id + '.bigwig')

    num_bp = np.array([23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352])
    chr_all = ['chr2L', 'chr2R', 'chr3L', 'chr3R', 'chr4', 'chrX', 'chrY']

    chr_len = {}
    for i in np.arange(len(chr_all)):
        chr_len[chr_all[i]] = num_bp[i]

    size = flanking_region_len*2  # 10240
    num_channel = 5

    genome_seq_batch = []
    label_batch = []

    feature_bw = pyBigWig.open(path2 + 'wt-ATAC.1x.P9108.bw')
    label_bw = pyBigWig.open(path3 + the_tf + peak_method + '.bw')
    tss_bed = open(path4, 'r')

    for idx, line in enumerate(tss_bed):

        # ignore header
        if idx == 0:
            continue

        bed_record = line.split("\t")
        the_chr = bed_record[0]
        start = int(bed_record[1])
        end = int(bed_record[2])

        # ignore chrom not in the chrom list
        if the_chr not in chr_all:
            continue

        if abs(start - end) != 2*flanking_region_len:
            print("start - end != flanking_region_len")
            continue

        label_bw_list = []

        # Error handling
        try:
            label_bw_list.append(np.array(label_bw.values(the_chr, start, end)).T)
        except RuntimeError:
            print(the_chr, start, end)
            print("Invalid interval bounds!")
            continue


        genome_seq = np.zeros((num_channel, size))
        num = 0
        for k in np.arange(len(list_dna)):
            the_id = list_dna[k]
            genome_seq[num, :] = dict_dna[the_id].values(the_chr, start, end)
            num += 1
        genome_seq[num, :] = np.nan_to_num(np.array(feature_bw.values(the_chr, start, end)), nan=0.0)
        # genome_seq[num+1, :] = np.nan_to_num(np.array(feature2_bw.values(the_chr, start, end)), nan=0.0)

        genome_seq_batch.append(genome_seq.T)
        label = np.stack(label_bw_list, axis=-1)
        label_batch.append(label)

    feature_bw.close()
    label_bw.close()
    tss_bed.close()
    return genome_seq_batch, label_batch

def generate_multi_data_batch(the_tf=None, peak_method=None, flanking_region_len=1000, file_path='./'):
    path1 = file_path+'/data/dna_bigwig/'  # dna
    path2 = file_path+'./data/atac_bigwig/'  # atac
    # path4 = './data/hg38.phyloP100way.bw' # conservation score
    
    if peak_method != "_2.5.fimo_labels":
        path3 = file_path+'/data/chipseq_6.5_bigwig/'  # label
    else:
        path3 = file_path+'/data/chipseq_bigwig/'  # label
        
    if flanking_region_len == 1000:
        path4 = file_path+'./data/dm6_refGene_Prom_Flanking_1000.bed'
    else:
        path4 = file_path+'./data/dm6_refGene_Prom_Flanking_10k.bed'
    # open bigwig
    list_dna = ['A', 'C', 'G', 'T']
    dict_dna = {}
    for the_id in list_dna:
        dict_dna[the_id] = pyBigWig.open(path1 + the_id + '.bigwig')

    num_bp = np.array([23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352])
    chr_all = ['chr2L', 'chr2R', 'chr3L', 'chr3R', 'chr4', 'chrX', 'chrY']

    chr_len = {}
    for i in np.arange(len(chr_all)):
        chr_len[chr_all[i]] = num_bp[i]

    size = flanking_region_len*2
    num_channel = 5

    genome_seq_batch = []
    label_batch = []

    feature_bw = pyBigWig.open(path2 + 'wt-ATAC.1x.P9108.bw')
    # label_bw = pyBigWig.open(path3 + the_tf + peak_method + '.bw')
    tss_bed = open(path4, 'r')

    for idx, line in enumerate(tss_bed):

        # ignore header
        if idx == 0:
            continue

        bed_record = line.split("\t")
        the_chr = bed_record[0]
        start = int(bed_record[1])
        end = int(bed_record[2])

        # ignore chrom not in the chrom list
        if the_chr not in chr_all:
            continue

        if abs(start - end) != 2*flanking_region_len:
            print("start - end != flanking_region_len*2")
            continue

        label_bw_list = []
        for single_tf in the_tf:
            label_bw = pyBigWig.open(path3 + single_tf + peak_method + '.bw')
            # Error handling
            try:
                label_bw_list.append(np.array(label_bw.values(the_chr, start, end)).T)
            except RuntimeError:
                print("Invalid interval bounds!")
                continue

        # if sum(label_bw_list[0]) == 0:
        #     print(sum(label_bw_list[0]))
        #     continue

        genome_seq = np.zeros((num_channel, size))
        num = 0
        for k in np.arange(len(list_dna)):
            the_id = list_dna[k]
            genome_seq[num, :] = dict_dna[the_id].values(the_chr, start, end)
            num += 1
        genome_seq[num, :] = np.nan_to_num(np.array(feature_bw.values(the_chr, start, end)), nan=0.0)
        # genome_seq[num+1, :] = np.nan_to_num(np.array(feature2_bw.values(the_chr, start, end)), nan=0.0)

        genome_seq_batch.append(genome_seq.T)
        label = np.stack(label_bw_list, axis=-1)
        label_batch.append(label)

    feature_bw.close()
    label_bw.close()
    tss_bed.close()
    return genome_seq_batch, label_batch

# def generate_balanced_data_batch(if_train=True, if_test=False, the_tf=None, peak_method=None, random_seed=30):
#     path1 = './data/dna_bigwig/'  # dna
#     path2 = './data/atac_bigwig/'  # atac
#     # path4 = './data/hg38.phyloP100way.bw' # conservation score
#     path3 = './data/chipseq_bigwig/'  # label
#
#     # open bigwig
#     list_dna = ['A', 'C', 'G', 'T']
#     dict_dna = {}
#     for the_id in list_dna:
#         dict_dna[the_id] = pyBigWig.open(path1 + the_id + '.bigwig')
#
#     num_bp = np.array([23513712, 25286936, 28110227, 32079331, 1348131, 23542271, 3667352])
#     chr_all = ['chr2L', 'chr2R', 'chr3L', 'chr3R', 'chr4', 'chrX', 'chrY']
#
#     chr_len = {}
#     for i in np.arange(len(chr_all)):
#         chr_len[chr_all[i]] = num_bp[i]
#
#     size = 2 ** 11 * 5  # 10240
#     num_channel = 5
#
#     num_sample = 10000
#     batch_size = 100
#
#     chr_train_all = ['chr2L', 'chr3L', 'chr3R', 'chr4', 'chrX', 'chrY']
#     ratio = 0.5
#     np.random.seed(random_seed)
#     np.random.shuffle(chr_train_all)
#     tmp = int(len(chr_train_all) * ratio)
#     chr_set1 = chr_train_all[:tmp]
#     chr_set2 = chr_train_all[tmp:]
#     chr_set3 = ['chr2R']
#     print(chr_set1)
#     print(chr_set2)
#     print(chr_set3)
#
#     index_set1 = random_shuffle(chr_set1, chr_len)
#     index_set2 = random_shuffle(chr_set2, chr_len)
#     index_set3 = random_shuffle(chr_set3, chr_len)
#
#     i = 0
#     pos_counter = 0
#     neg_counter = 0
#     genome_seq_batch = []
#     label_batch = []
#
#     # steps_per_epoch=int(num_sample // batch_size)
#     while pos_counter < int(num_sample*0.5):
#         if if_test:
#             if i == len(index_set3):
#                 i = 0
#                 np.random.shuffle(index_set3)
#             the_chr = index_set3[i]
#             i += 1
#         elif if_train:
#             if i == len(index_set1):
#                 i = 0
#                 np.random.shuffle(index_set1)
#             the_chr = index_set1[i]
#             i += 1
#         else:
#             if i == len(index_set2):
#                 i = 0
#                 np.random.shuffle(index_set2)
#             the_chr = index_set2[i]
#             i += 1
#
#         feature_bw = pyBigWig.open(path2 + 'wt-ATAC.1x.P9108.bw')
#
#         start = int(np.random.randint(0, chr_len[the_chr] - size, 1))
#         end = start + size
#         # print(the_chr, " START: ", start, " END: ", end)
#
#         label_bw_list = []
#         # for tf in the_tf:
#         label_bw = pyBigWig.open(path3 + the_tf + peak_method + '.bw')
#         label_tmp = np.nan_to_num(np.array(label_bw.values(the_chr, start, end)).T, nan=0.0)
#         # label_bw_list.append(np.array(label_bw.values(the_chr, start, end)).T)
#
#         if sum(label_tmp) > 0:
#             pos_counter += 1
#             label_bw_list.append(label_tmp)
#             genome_seq = np.zeros((num_channel, size))
#             num = 0
#             for k in np.arange(len(list_dna)):
#                 the_id = list_dna[k]
#                 genome_seq[num, :] = dict_dna[the_id].values(the_chr, start, end)
#                 num += 1
#             genome_seq[num, :] = np.nan_to_num(np.array(feature_bw.values(the_chr, start, end)), nan=0.0)
#
#             genome_seq_batch.append(genome_seq.T)
#             label = np.stack(label_bw_list, axis=-1)
#             label_batch.append(label)
#         elif neg_counter < int(num_sample*0.5):
#             neg_counter += 1
#             label_bw_list.append(label_tmp)
#             genome_seq = np.zeros((num_channel, size))
#             num = 0
#             for k in np.arange(len(list_dna)):
#                 the_id = list_dna[k]
#                 genome_seq[num, :] = dict_dna[the_id].values(the_chr, start, end)
#                 num += 1
#             genome_seq[num, :] = np.nan_to_num(np.array(feature_bw.values(the_chr, start, end)), nan=0.0)
#
#             genome_seq_batch.append(genome_seq.T)
#             label = np.stack(label_bw_list, axis=-1)
#             label_batch.append(label)
#
#         label_bw.close()
#
#     return genome_seq_batch, label_batch


def generate_dataset(the_tf=None, peak_method=None,  random_seed=30, if_train=True, if_test=False, if_balance=False):
    if if_balance:
        pass
        # dataset = tf.data.Dataset.from_tensor_slices(
        #     generate_balanced_data_batch(if_train=if_train, if_test=if_test, the_tf=the_tf,
        #                                  peak_method=peak_method, random_seed=random_seed))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            generate_data_batch(if_train=if_train, if_test=if_test, the_tf=the_tf,
                                peak_method=peak_method, random_seed=random_seed))
    return dataset
