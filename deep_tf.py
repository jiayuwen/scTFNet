import argparse
from models import *
from utils.plot_utils import *
from utils.metrics_utils import *
from utils.train_utils import *
import horovod.tensorflow.keras as hvd
import time

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def get_args():
    parser = argparse.ArgumentParser(description="DeepTF")
    parser.add_argument('-tf', '--transcription_factor', default=['Ttk69'], type=str, nargs='+',
                        help='transcript factor')
    parser.add_argument('-m', '--models', default='trans_unet', type=str,
                        help='model architecture')
    # parser.add_argument('-l', '--input_length', default=2000, type=int,
    #                     help='length of input sequence')
    parser.add_argument('-p', '--peak_calling_method', default='_2.5.fimo_labels', type=str,
                        help='peak calling method')
    parser.add_argument('-v', '--vocab_size', default=5, type=int,
                        help='vocabulary size of the input')
    parser.add_argument('-n', '--flanking_region', default=1000, type=int,
                        help='length of flanking region (default using 1000 flanking region)')
    parser.add_argument('-b', '--batch_size', default=100, type=int,
                        help='number of samples in each batch (default batch_size = 100)')
    parser.add_argument('-d', '--draw_frequency', default=1, type=int,
                        help='draw frequency (default draw 1 time)')
    parser.add_argument('-r', '--random_seed', default=1, type=int,
                        help='fix random seed')
    parser.add_argument('-f', '--path', default='./', type=str,
                        help='save data to path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    the_tf = args.transcription_factor
    batch_size = args.batch_size
    peak_method = args.peak_calling_method
    flanking_region_len = args.flanking_region
    max_len = flanking_region_len*2
    vocab_size = args.vocab_size
    model_arch = args.models
    draw_times = args.draw_frequency
    file_path = args.path

    print(args)

    train_data_name = "train_data"
    val_data_name = "validation_data"
    test_data_name = "test_data"
    if len(the_tf) > 1:
        data_dir = file_path + "preprocessed_data/" + "preprocessed_" + 'multi' + peak_method + "_fimo_data/"
    else:
        data_dir = file_path + "preprocessed_data/" + "preprocessed_" + ''.join(the_tf) + peak_method + "_fimo_data/"
    if len(the_tf) > 1:
        output_dir = file_path + 'output/' + 'multi' + peak_method + '_' + model_arch + "_output/"
    else:
        output_dir = file_path + 'output/' + ''.join(the_tf) + peak_method + '_' + model_arch + "_output/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = output_dir + model_arch + '_checkpoints/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    model_dir = output_dir + 'model/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    plot_dir = output_dir + 'plot/'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    result_dir = output_dir + 'result/'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # data preprocessing
    # os.system(f"python data_preprocessing.py -tf {the_tf} -f {flanking_region_len} -m {peak_method} -p {file_path}")

    # load model
    model = return_model(model_arch, max_len, vocab_size, tf_num=len(the_tf))

    opt = tf.optimizers.Adam(0.001 * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc"),
                           tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc"),
                           f1_metrics,
                           dice_coef,
                           matthews_correlation_coefficient,
                           tf.keras.metrics.BinaryIoU([1], threshold=0.5)])

    model.summary()

    scheduler = CosineScheduler(max_update=100, base_lr=0.001*hvd.size(), final_lr=0.0001*hvd.size(), warmup_steps=10, warmup_begin_lr=0.0001)
    # scheduler = CosineScheduler(max_update=100, base_lr=0.0005*hvd.size(), final_lr=0.00001*hvd.size(), warmup_steps=20, warmup_begin_lr=0.00001)
    learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        learning_rate,
        early_stop]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    # if hvd.rank() == 0:
    #     callbacks.append(
    #         tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '-{epoch}.h5', save_freq=5 * batch_size * 2, verbose=1))

    train_dataset = tf.data.experimental.load(os.path.join(data_dir + train_data_name))
    val_dataset = tf.data.experimental.load(os.path.join(data_dir + val_data_name))
    test_dataset = tf.data.experimental.load(os.path.join(data_dir + test_data_name))

    train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
    val_dataset = val_dataset.shard(hvd.size(), hvd.rank())
    test_dataset = test_dataset.shard(hvd.size(), hvd.rank())

    train_dataset = train_dataset.shuffle(2000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)
    
    tik = time.time()
    history = model.fit(x=train_dataset, epochs=200, batch_size=batch_size,
                        validation_data=val_dataset, callbacks=callbacks)
    tok = time.time()

    print(plot_pr_auc(history, path=plot_dir))

    # only 1 GPU will need to do following task
    if hvd.rank() == 0:
        # save model
        model.save(model_dir)
        # model evaluation
        result_dic = model.evaluate(test_dataset, batch_size=batch_size, return_dict=True)
        result_dic["training_time"] = tok - tik
        save_result(result_dic, result_dir)


if __name__ == '__main__':
    main()
