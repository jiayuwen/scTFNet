import argparse
from models import *
from utils.plot_utils import *
from utils.metrics_utils import *
from utils.train_utils import *
import horovod.tensorflow.keras as hvd
import time
import pyBigWig
import tensorflow_addons as tfa
import keras_tuner

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
        
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads=8, ff_dim=64, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn
        })
        return config

class PositionEmbedding(Layer):
    def __init__(self, max_len):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len

    def call(self, x):
        positions = tf.range(start=0, limit=2000, delta=2000/self.max_len)
        positions = tf.math.sin(positions)
        return tf.math.multiply(x, tf.expand_dims(positions, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len
        })
        return config


def EncoderSeTBlock(inputs, pos_len, num_heads=8, num_blocks=0, n_filters=32, kernel_size=11, dropout_prob=0.3, ratio=2, layer_name="", max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width
    # (hence, is not reduced in size)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(se, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink
    # the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    
    # Pooling reduces the size of the image while keeping the number of channels same Pooling has been kept as
    # optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use) Below,
    # Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse
    # across input image
    if max_pooling:
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    else:
        x = conv
    
    if num_blocks>0:
        embedding_layer = PositionEmbedding(pos_len)
        
        transformer_block_list = []
        for i in np.arange(num_blocks):
            transformer_block_list.append(TransformerBlock(embed_dim=n_filters, num_heads=num_heads))
        x = embedding_layer(x)
        for t_block in transformer_block_list:
            x = t_block(x)
    
    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
    # transpose convolutions
    skip_connection = conv

    return x, skip_connection


def DecoderSeTBlock(prev_layer_input, skip_layer_input, pos_len, num_heads=8, num_blocks=0, kernel_size=7, ratio=2, layer_name="", n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv1DTranspose(#input_tensor=prev_layer_input,
                         filters=n_filters,
                         kernel_size=kernel_size,  # Kernel size
                         strides=2,
                         padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=-1)
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
    x = BatchNormalization()(se)
    
    if num_blocks>0:
        embedding_layer = PositionEmbedding(pos_len)
        transformer_block_list = []
        for i in np.arange(num_blocks):
            transformer_block_list.append(TransformerBlock(embed_dim=n_filters, num_heads=num_heads))
        x = embedding_layer(x)
        for t_block in transformer_block_list:
            x = t_block(x)
    
    return x

def TUNet(hparams, input_size=(2000, 5), n_filters=32, n_classes=14):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)
    
    # Hyper Parameters
    # num_units = hparams[HP_NUM_UNITS]
    kernel_size = hparams.Int("kernel_size", min_value=3, max_value=11, step=2)
    dropout_prob = hparams.Float("dropout", min_value=0.1, max_value=0.3, step=0.1)
    num_heads = hparams.Int("num_heads", min_value=4, max_value=12, step=4)
    num_blocks = hparams.Int("num_blocks", min_value=0, max_value=12, step=4)
    
    cblock1 = EncoderSeTBlock(inputs, pos_len=2000, num_heads=0, n_filters=32, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_1_", max_pooling=True)
    cblock2 = EncoderSeTBlock(cblock1[0], pos_len=1000, num_heads=0, n_filters=64, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_2_", max_pooling=True)
    cblock3 = EncoderSeTBlock(cblock2[0], pos_len=500, num_heads=0, n_filters=128, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_3_", max_pooling=True)
    cblock4 = EncoderSeTBlock(cblock3[0], pos_len=250,  num_heads=0, n_filters=256, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_4_", max_pooling=True)
    cblock5 = EncoderSeTBlock(cblock4[0], pos_len=125,  num_heads=num_heads, num_blocks=num_blocks, n_filters=256, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_5_", max_pooling=False)

    ublock6 = DecoderSeTBlock(cblock5[0], cblock4[1], pos_len=250, num_heads=num_heads, kernel_size=kernel_size, n_filters=256, layer_name="dcb_1_")
    ublock7 = DecoderSeTBlock(ublock6, cblock3[1], pos_len=500, num_heads=num_heads, kernel_size=kernel_size, n_filters=128, layer_name="dcb_2_")
    ublock8 = DecoderSeTBlock(ublock7, cblock2[1], pos_len=1000, num_heads=0, kernel_size=kernel_size, n_filters=128, layer_name="dcb_3_")
    ublock9 = DecoderSeTBlock(ublock8, cblock1[1], pos_len=2000, num_heads=0, kernel_size=kernel_size, n_filters=128, layer_name="dcb_4_")
    conv9_0 = Conv1D(filters=128,
                   kernel_size=11,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)
    x = SqueezeExcitation1DLayer(out_dim=128, ratio=8, layer_name='se_0')(conv9_0)
    conv9_1 = Conv1D(filters=64,
                   kernel_size=7,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(x)
    x = SqueezeExcitation1DLayer(out_dim=64, ratio=8, layer_name='se_1')(conv9_1)
    conv9_2 = Conv1D(filters=32,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(x)
    x = SqueezeExcitation1DLayer(out_dim=32, ratio=8, layer_name='se_2')(conv9_2)
    dense1_0 = Dense(32, activation='relu')(x)
    dense1_1 = Dense(16, activation='relu')(dense1_0)
    dense1_2 = Dense(n_classes, activation='sigmoid')(dense1_1)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=dense1_2)
    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=[tf.keras.metrics.AUC(curve="ROC", num_thresholds=1001, name="auc"),
               tf.keras.metrics.AUC(curve="PR", num_thresholds=1001, name="pr_auc"),
               tf.keras.metrics.BinaryCrossentropy(
    name="binary_crossentropy", dtype=None, from_logits=False, label_smoothing=0
),
               f1_metrics,
               dice_coef,
               matthews_correlation_coefficient,
               tf.keras.metrics.BinaryIoU([1], threshold=0.5)]
    )
    return model

batch_size = 100*int(len(gpus)) #global batch size
train_data_name = "train_data"
val_data_name = "validation_data"
test_data_name = "test_data"
data_dir = "preprocessed_data/preprocessed_multi_2.5.fimo_labels_fimo_data/"

train_dataset = tf.data.experimental.load(os.path.join(data_dir + train_data_name))
val_dataset = tf.data.experimental.load(os.path.join(data_dir + val_data_name))
test_dataset = tf.data.experimental.load(os.path.join(data_dir + test_data_name))

train_dataset = train_dataset.shuffle(2000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

# # Hyperband
# tuner = keras_tuner.Hyperband(
#     hypermodel=TUNet,
#     objective=keras_tuner.Objective("val_pr_auc", direction="max"),
#     max_epochs=100,
#     factor=3,
#     hyperband_iterations=10,
#     distribution_strategy=tf.distribute.MirroredStrategy(),
#     directory="logs",
#     project_name="keras_tuner_hyperband",
#     seed=30,
#     overwrite=True
# )

#Random Search
# tuner = keras_tuner.RandomSearch(
#     hypermodel=TUNet,
#     objective=keras_tuner.Objective("val_pr_auc", direction="max"),
#     max_trials=10,
#     executions_per_trial=3,
#     distribution_strategy=tf.distribute.MirroredStrategy(),
#     directory="logs",
#     project_name="keras_tuner_random_search",
#     seed=30,
#     overwrite=True,
# )

#bayesian
tuner = keras_tuner.BayesianOptimization(
    hypermodel=TUNet,
    objective=keras_tuner.Objective("val_pr_auc", direction="max"),
    max_trials=10,
    executions_per_trial=3,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory="logs",
    project_name="keras_tuner_bayesian",
    seed=30,
    overwrite=True,
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    x=train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks = [early_stop,tf.keras.callbacks.TensorBoard("logs/keras_tuner_bayesian/tensorboard/")]
)

tuner.results_summary()