import argparse
from utils.data_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Transcription Factor Data Preprocessing")
    parser.add_argument('-tf', '--transcription_factor', default='ttk69', type=str,
                        help='transcript factor')
    parser.add_argument('-m', '--peak_calling_method', default='_6.5.fimo_labels', type=str,
                        help='peak calling method')
    parser.add_argument('-b', '--balanced_dataset', default=False, type=bool,
                        help='balanced_dataset option')
    parser.add_argument('-f', '--flanking_region', default=1000, type=int,
                        help='length of flanking region (default using 1000 flanking region)')
    parser.add_argument('-d', '--draw_frequency', default=1, type=int,
                        help='draw frequency (default draw 1 time)')
    parser.add_argument('-p', '--path', default='./', type=str,
                        help='save data to path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    the_tf = args.transcription_factor
    peak_method = args.peak_calling_method
    if_balance = args.balanced_dataset
    flanking_region_len = args.flanking_region
    file_path = args.path

    if if_balance:
        data_dir = file_path + "preprocessed_data/" + "preprocessed_balanced_" + the_tf + "_fimo_data/"
    else:
        data_dir = file_path + "preprocessed_data/" + "preprocessed_" + the_tf + peak_method +"_f"+ str(flanking_region_len)+"_fimo_data/"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    full_dataset = tf.data.Dataset.from_tensor_slices(
        generate_data_batch(the_tf=the_tf, peak_method=peak_method, flanking_region_len=flanking_region_len, file_path=file_path))

    DATASET_SIZE = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)
    print(train_size, val_size, test_size)

    full_dataset = full_dataset.shuffle(50000, reshuffle_each_iteration=False)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    print("TRAINING DATA")
    tf.data.experimental.save(train_dataset, os.path.join(data_dir + "train_data"))
    del train_dataset

    print("VALIDATION DATA")
    tf.data.experimental.save(val_dataset, os.path.join(data_dir + "validation_data"))
    del val_dataset

    print("TEST DATA")
    tf.data.experimental.save(test_dataset, os.path.join(data_dir + "test_data"))
    del test_dataset

if __name__ == '__main__':
    main()