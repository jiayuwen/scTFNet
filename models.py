import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, \
    BatchNormalization, Bidirectional, LSTM, Dropout, Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
    AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling1D, MultiHeadAttention,\
    LayerNormalization, Embedding, LeakyReLU, Conv1DTranspose
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np

def return_model(model_name, max_len, vocab_size, tf_num):
    model_dic={#'se_cnn': cnn_model(max_len, vocab_size, tf_num),
               'unet': UNet(input_size=(max_len, vocab_size), n_classes=tf_num),
               'se_unet': TUNet(input_size=(max_len, vocab_size), n_classes=tf_num, num_blocks = 0),
               'trans_unet': TUNet(input_size=(max_len, vocab_size), n_classes=tf_num)}
    return model_dic[model_name]


def EncoderBlock(inputs, n_filters=32, kernel_size=11, dropout_prob=0.1, max_pooling=True):
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
    bn = BatchNormalization()(conv)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink
    # the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same Pooling has been kept as
    # optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use) Below,
    # Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse
    # across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
    # transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection

def DecoderBlock(prev_layer_input, skip_layer_input, kernel_size=11, n_filters=32):
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
    bn = BatchNormalization()(conv)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    bn = BatchNormalization()(conv)
    return bn

def UNet(input_size=(2000, 5), n_filters=32, n_classes=1):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of
    # the image
    cblock1 = EncoderBlock(inputs, n_filters, dropout_prob=0.1, max_pooling=True)
    cblock2 = EncoderBlock(cblock1[0], 64, dropout_prob=0.1, max_pooling=True)
    cblock3 = EncoderBlock(cblock2[0], 128, dropout_prob=0.1, max_pooling=True)
    cblock4 = EncoderBlock(cblock3[0], 256, dropout_prob=0.1, max_pooling=True)
    cblock5 = EncoderBlock(cblock4[0], 256, dropout_prob=0.1, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderBlock(cblock5[0], cblock4[1], kernel_size=11, n_filters=256)
    ublock7 = DecoderBlock(ublock6, cblock3[1], kernel_size=11, n_filters=128)
    ublock8 = DecoderBlock(ublock7, cblock2[1], kernel_size=11, n_filters=128)
    ublock9 = DecoderBlock(ublock8, cblock1[1], kernel_size=11, n_filters=128)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv1D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv1D(n_classes, 1, activation='sigmoid', padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

class SqueezeExcitation1DLayer(tf.keras.Model):

    def __init__(self, out_dim, ratio, layer_name="se"):
        super(SqueezeExcitation1DLayer, self).__init__(name=layer_name)
        self.squeeze = GlobalAveragePooling1D()
        self.excitation_a = Dense(units=out_dim / ratio, activation='relu')
        self.excitation_b = Dense(units=out_dim, activation='sigmoid')
        self.shape = [-1, 1, out_dim]

    def call(self, input_x):
        squeeze = self.squeeze(input_x)

        excitation = self.excitation_a(squeeze)
        excitation = self.excitation_b(excitation)

        scale = tf.reshape(excitation, self.shape)
        se = input_x * scale

        return se

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

# UNet adopted from https://github.com/VidushiBhatia/U-Net-Implementation

def EncoderSeTBlock(inputs, pos_len, num_heads=8, num_blocks=0, n_filters=32, kernel_size=11, dropout_prob=0.3, ratio=2, layer_name="", max_pooling=True):
    """
    This block uses multiple convolution layers and squeeze excitation layers, which followed by max pool, relu activation to create an architecture for learning.
    Optional transformer encoder block can learn features in a long range.
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


def TUNet(input_size=(2000, 5), n_classes=14, num_blocks = 12):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)
    
    # Hyper Parameters
    # num_units = hparams[HP_NUM_UNITS]
    # kernel_size = hparams.Int("kernel_size", min_value=3, max_value=11, step=2)
    # dropout_prob = hparams.Float("dropout", min_value=0.1, max_value=0.3, step=0.1)
    # num_heads = hparams.Int("num_heads", min_value=0, max_value=12, step=2)
    kernel_size = 11
    dropout_prob = 0.1
    # num_blocks = 12
    num_heads = 8
    
    cblock1 = EncoderSeTBlock(inputs, pos_len=2000, num_heads=0, n_filters=32, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_1_", max_pooling=True)
    cblock2 = EncoderSeTBlock(cblock1[0], pos_len=1000, num_heads=0, n_filters=64, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_2_", max_pooling=True)
    cblock3 = EncoderSeTBlock(cblock2[0], pos_len=500, num_heads=0, n_filters=128, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_3_", max_pooling=True)
    cblock4 = EncoderSeTBlock(cblock3[0], pos_len=250,  num_heads=0, n_filters=256, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_4_", max_pooling=True)
    cblock5 = EncoderSeTBlock(cblock4[0], pos_len=125,  num_heads=num_heads, num_blocks=num_blocks, n_filters=256, kernel_size=kernel_size, dropout_prob=dropout_prob, layer_name="ecb_5_", max_pooling=False)

    ublock6 = DecoderSeTBlock(cblock5[0], cblock4[1], pos_len=250, num_heads=0, kernel_size=kernel_size, n_filters=256, layer_name="dcb_1_")
    ublock7 = DecoderSeTBlock(ublock6, cblock3[1], pos_len=500, num_heads=0, kernel_size=kernel_size, n_filters=128, layer_name="dcb_2_")
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
    return model
#########################################################################################################
# outdated code below
#########################################################################################################



# def cnn_model(max_len, vocab_size, tf_num):
#     model = Sequential([
#         InputLayer(input_shape=(max_len, vocab_size)),
#         Conv1D(32, 11, padding='same', activation='relu'),
#         BatchNormalization(),
#         LayerNormalization(),
#         SqueezeExcitation1DLayer(out_dim=32, ratio=2, layer_name='se0'),
#         Dropout(0.5),
#         Conv1D(64, 11, padding='same', activation='relu'),
#         BatchNormalization(),
#         LayerNormalization(),
#         SqueezeExcitation1DLayer(out_dim=64, ratio=4, layer_name='se1'),
#         Dropout(0.5),
#         Conv1D(128, 11, padding='same', activation='relu'),
#         BatchNormalization(),
#         LayerNormalization(),
#         SqueezeExcitation1DLayer(out_dim=128, ratio=8, layer_name='se2'),
#         Conv1D(128, 3, padding='same', activation='relu'),
#         SqueezeExcitation1DLayer(out_dim=128, ratio=8),
#         Conv1D(tf_num, 3, padding='same', activation='sigmoid')
#         # Dense(tf_num, activation='sigmoid')
#     ])
#     return model
# def unet(max_len, vocab_size, tf_num=1):
#     model = UNet(input_size=(max_len, vocab_size), n_classes=tf_num)
#     return model

# def EncoderBlock(inputs, n_filters=32, kernel_size=11, dropout_prob=0.3, max_pooling=True):
#     """
#     This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
#     Dropout can be added for regularization to prevent overfitting.
#     The block returns the activation values for next layer along with a skip connection which will be used in the decoder
#     """
#     # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
#     # Proper initialization prevents from the problem of exploding and vanishing gradients
#     # 'Same' padding will pad the input to conv layer such that the output has the same height and width
#     # (hence, is not reduced in size)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(inputs)
#     bn = BatchNormalization()(conv)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(bn)

#     # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
#     conv = BatchNormalization()(conv, training=False)

#     # In case of overfitting, dropout will regularize the loss and gradient computation to shrink
#     # the influence of weights on output
#     if dropout_prob > 0:
#         conv = tf.keras.layers.Dropout(dropout_prob)(conv)

#     # Pooling reduces the size of the image while keeping the number of channels same Pooling has been kept as
#     # optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use) Below,
#     # Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse
#     # across input image
#     if max_pooling:
#         next_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
#     else:
#         next_layer = conv

#     # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
#     # transpose convolutions
#     skip_connection = conv

#     return next_layer, skip_connection


# def DecoderBlock(prev_layer_input, skip_layer_input, kernel_size=11, n_filters=32):
#     """
#     Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
#     merges the result with skip layer results from encoder block
#     Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
#     The function returns the decoded layer output
#     """
#     # Start with a transpose convolution layer to first increase the size of the image
#     up = Conv1DTranspose(#input_tensor=prev_layer_input,
#                          filters=n_filters,
#                          kernel_size=kernel_size,  # Kernel size
#                          strides=2,
#                          padding='same')(prev_layer_input)

#     # Merge the skip connection from previous block to prevent information loss
#     merge = concatenate([up, skip_layer_input], axis=-1)

#     # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
#     # The parameters for the function are similar to encoder
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(merge)
#     bn = BatchNormalization()(conv)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(bn)
#     bn = BatchNormalization()(conv)
#     return bn


# def UNet(input_size=(2000, 5), n_filters=32, n_classes=1):
#     """
#     Combine both encoder and decoder blocks according to the U-Net research paper
#     Return the model as output
#     """
#     # Input size represent the size of 1 image (the size used for pre-processing)
#     inputs = Input(input_size)

#     # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
#     # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of
#     # the image
#     cblock1 = EncoderBlock(inputs, n_filters, dropout_prob=0.1, max_pooling=True)
#     cblock2 = EncoderBlock(cblock1[0], 64, dropout_prob=0.1, max_pooling=True)
#     cblock3 = EncoderBlock(cblock2[0], 128, dropout_prob=0.1, max_pooling=True)
#     cblock4 = EncoderBlock(cblock3[0], 256, dropout_prob=0.1, max_pooling=True)
#     cblock5 = EncoderBlock(cblock4[0], 256, dropout_prob=0.1, max_pooling=False)

#     # Decoder includes multiple mini blocks with decreasing number of filters
#     # Observe the skip connections from the encoder are given as input to the decoder
#     # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
#     ublock6 = DecoderBlock(cblock5[0], cblock4[1], kernel_size=11, n_filters=256)
#     ublock7 = DecoderBlock(ublock6, cblock3[1], kernel_size=11, n_filters=128)
#     ublock8 = DecoderBlock(ublock7, cblock2[1], kernel_size=11, n_filters=64)
#     ublock9 = DecoderBlock(ublock8, cblock1[1], kernel_size=11, n_filters=32)

#     # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
#     # Followed by a 1x1 Conv layer to get the image to the desired size.
#     # Observe the number of channels will be equal to number of output classes
#     conv9 = Conv1D(n_filters,
#                    3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(ublock9)

#     conv10 = Conv1D(n_classes, 1, activation='sigmoid', padding='same')(conv9)

#     # Define the model
#     model = tf.keras.Model(inputs=inputs, outputs=conv10)

#     return model

# class TransformerBlock(Layer):
#     def __init__(self, embed_dim, num_heads=8, ff_dim=64, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = Sequential(
#             [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
#         )
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output) # residual connection
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output) # residual connection

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "att": self.att,
#             "ffn": self.ffn
#         })
#         return config

# class PositionEmbedding(Layer):
#     def __init__(self, max_len):
#         super(PositionEmbedding, self).__init__()
#         self.max_len = max_len

#     def call(self, x):
#         positions = tf.range(start=0, limit=200, delta=200/self.max_len)
#         positions = tf.math.sin(positions)
#         return tf.math.multiply(x, tf.expand_dims(positions, axis=-1))

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "max_len": self.max_len
#         })
#         return config


# def EncoderSeTBlock(inputs, pos_len, n_filters=32, kernel_size=11, dropout_prob=0.3, ratio=2, layer_name="", max_pooling=True):
#     """
#     This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
#     Dropout can be added for regularization to prevent overfitting.
#     The block returns the activation values for next layer along with a skip connection which will be used in the decoder
#     """
#     # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
#     # Proper initialization prevents from the problem of exploding and vanishing gradients
#     # 'Same' padding will pad the input to conv layer such that the output has the same height and width
#     # (hence, is not reduced in size)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(inputs)
#     se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
#     bn = BatchNormalization()(se)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(bn)
#     se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
#     # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
#     conv = BatchNormalization()(se, training=False)

#     # In case of overfitting, dropout will regularize the loss and gradient computation to shrink
#     # the influence of weights on output
#     if dropout_prob > 0:
#         conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    
#     embedding_layer = PositionEmbedding(pos_len)
#     transformer_block_1 = TransformerBlock(embed_dim=n_filters)
#     transformer_block_2 = TransformerBlock(embed_dim=n_filters)
    
#     x = embedding_layer(conv)
#     x = transformer_block_1(x)
#     conv = transformer_block_2(x)
    
#     # Pooling reduces the size of the image while keeping the number of channels same Pooling has been kept as
#     # optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use) Below,
#     # Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse
#     # across input image
#     if max_pooling:
#         next_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
#     else:
#         next_layer = conv

#     # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
#     # transpose convolutions
#     skip_connection = conv

#     return next_layer, skip_connection


# def DecoderSeTBlock(prev_layer_input, skip_layer_input, pos_len, kernel_size=7, ratio=2, layer_name="", n_filters=32):
#     """
#     Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
#     merges the result with skip layer results from encoder block
#     Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
#     The function returns the decoded layer output
#     """
#     # Start with a transpose convolution layer to first increase the size of the image
#     up = Conv1DTranspose(#input_tensor=prev_layer_input,
#                          filters=n_filters,
#                          kernel_size=kernel_size,  # Kernel size
#                          strides=2,
#                          padding='same')(prev_layer_input)

#     # Merge the skip connection from previous block to prevent information loss
#     merge = concatenate([up, skip_layer_input], axis=-1)
#     # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
#     # The parameters for the function are similar to encoder
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(merge)
#     se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
#     bn = BatchNormalization()(se)
#     conv = Conv1D(n_filters,
#                   kernel_size,  # Kernel size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(bn)
#     se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
#     bn = BatchNormalization()(se)
    
#     embedding_layer = PositionEmbedding(pos_len)
#     transformer_block_1 = TransformerBlock(embed_dim=n_filters)
#     transformer_block_2 = TransformerBlock(embed_dim=n_filters)
    
#     x = embedding_layer(bn)
#     x = transformer_block_1(x)
#     x = transformer_block_2(x)
    
#     return x

# def TUNet(input_size=(2000, 5), n_classes=1):
#     """
#     Combine both encoder and decoder blocks according to the U-Net research paper
#     Return the model as output
#     """
#     # Input size represent the size of 1 image (the size used for pre-processing)
#     inputs = Input(input_size)
#     pos_len = input_size[0]
#     cblock1 = EncoderSeTBlock(inputs, pos_len=pos_len, n_filters=32, kernel_size=11, dropout_prob=0.1, layer_name="ecb_1_", max_pooling=True)
#     cblock2 = EncoderSeTBlock(cblock1[0], pos_len=pos_len//2, n_filters=64, kernel_size=11, dropout_prob=0.1, layer_name="ecb_2_", max_pooling=True)
#     cblock3 = EncoderSeTBlock(cblock2[0], pos_len=pos_len//4, n_filters=128, kernel_size=11, dropout_prob=0.1, layer_name="ecb_3_", max_pooling=True)
#     cblock4 = EncoderSeTBlock(cblock3[0], pos_len=pos_len//8, n_filters=256, kernel_size=11, dropout_prob=0.1, layer_name="ecb_4_", max_pooling=True)
#     cblock5 = EncoderSeTBlock(cblock4[0], pos_len=pos_len//16, n_filters=256, kernel_size=11, dropout_prob=0.1, layer_name="ecb_5_", max_pooling=False)
    
#     ublock6 = DecoderSeTBlock(cblock5[0], cblock4[1], pos_len=pos_len//8, kernel_size=11, n_filters=256, layer_name="dcb_1_")
#     ublock7 = DecoderSeTBlock(ublock6, cblock3[1], pos_len=pos_len//4, kernel_size=11, n_filters=128, layer_name="dcb_2_")
#     ublock8 = DecoderSeTBlock(ublock7, cblock2[1], pos_len=pos_len//2, kernel_size=11, n_filters=128, layer_name="dcb_3_")
#     ublock9 = DecoderSeTBlock(ublock8, cblock1[1], pos_len=pos_len, kernel_size=11, n_filters=128, layer_name="dcb_4_")
#     conv9_0 = Conv1D(filters=128,
#                    kernel_size=11,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(ublock9)
#     x = SqueezeExcitation1DLayer(out_dim=128, ratio=8, layer_name='se_0')(conv9_0)
#     conv9_1 = Conv1D(filters=64,
#                    kernel_size=7,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(x)
#     x = SqueezeExcitation1DLayer(out_dim=64, ratio=8, layer_name='se_1')(conv9_1)
#     conv9_2 = Conv1D(filters=32,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(x)
#     x = SqueezeExcitation1DLayer(out_dim=32, ratio=8, layer_name='se_2')(conv9_2)
#     dense1_0 = Dense(32, activation='relu')(x)
#     dense1_1 = Dense(16, activation='relu')(dense1_0)
#     dense1_2 = Dense(n_classes, activation='sigmoid')(dense1_1)

#     # Define the model
#     model = tf.keras.Model(inputs=inputs, outputs=dense1_2)

#     return model



#########################################################################################################
# outdated code below
#########################################################################################################
# def transformer_model1(max_len, vocab_size, batch_size=16, num_heads=4, ff_dim=64):
#     inputs = Input((2000, 5))
#     x_sc1 = Conv1D(64, 9, strides=2, padding='same', activation='relu')(inputs)
#     x = BatchNormalization()(x_sc1)
#     x = LayerNormalization()(x)
#     x = Conv1D(64, 6, strides=2, padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#     x_sc2 = LayerNormalization()(x)

#     transformer_block_1 = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)
#     transformer_block_2 = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)
#     transformer_block_3 = TransformerBlock(embed_dim=64, num_heads=num_heads * 2, ff_dim=ff_dim)
#     transformer_block_4 = TransformerBlock(64, num_heads * 2, ff_dim)
#     transformer_block_5 = TransformerBlock(64, num_heads * 2, ff_dim)
#     transformer_block_6 = TransformerBlock(64, num_heads * 2, ff_dim)

#     # embedding_layer = TokenAndPositionEmbedding(max_len//4, vocab_size, max_len)
#     # x = embedding_layer(x_sc2)
#     x = transformer_block_1(x_sc2)
#     x1 = transformer_block_2(x)
#     x = transformer_block_3(x1) + x1
#     x2 = transformer_block_4(x)
#     x = transformer_block_5(x2) + x2
#     x = transformer_block_6(x)
#     # x = Flatten()(x)

#     x = Conv1D(64, 17, padding='same', activation='relu')(x)
#     x = concatenate([x, x_sc2], axis=-1)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = SqueezeExcitation1DLayer(out_dim=64*2, ratio=4, layer_name='se1')(x)
#     x = Dropout(0.5)(x)
#     x = concatenate([x, x_sc2], axis=-1)
#     x = Conv1DTranspose(64, kernel_size=6, strides=2, padding='same')(x)
#     x = Conv1D(128, 5, padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = SqueezeExcitation1DLayer(out_dim=128, ratio=8, layer_name='se2')(x)
#     x = concatenate([x, x_sc1], axis=-1)
#     x = Conv1DTranspose(128, kernel_size=6, strides=2, padding='same')(x)
#     x = Conv1D(128, 3, padding='same', activation='relu')(x)
#     x = SqueezeExcitation1DLayer(out_dim=128, ratio=8)(x)
#     x = Dense(128)(x)
#     x = tf.keras.activations.relu(x, alpha=0.1)
#     x = Dropout(0.2)(x)
#     x = Dense(64)(x)
#     x = tf.keras.activations.relu(x, alpha=0.1)
#     x = Dropout(0.2)(x)
#     x = Dense(32)(x)
#     x = tf.keras.activations.relu(x, alpha=0.1)
#     x = Dropout(0.1)(x)
#     x = Dense(16)(x)
#     x = tf.keras.activations.relu(x, alpha=0.1)
#     x = Dropout(0.1)(x)
#     outputs = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# def transformer_model(max_len, vocab_size, batch_size=16, num_heads=4, ff_dim=64):

#     embedding_layer_1 = TokenAndPositionEmbedding(max_len, vocab_size, max_len)
#     embedding_layer_2 = TokenAndPositionEmbedding(max_len//2, vocab_size, max_len)
#     embedding_layer_3 = TokenAndPositionEmbedding(max_len//4, vocab_size, max_len)
#     embedding_layer_4 = TokenAndPositionEmbedding(max_len//4, vocab_size, max_len)
#     embedding_layer_5 = TokenAndPositionEmbedding(max_len//2, vocab_size, max_len)
#     embedding_layer_6 = TokenAndPositionEmbedding(max_len, vocab_size, max_len)
#     transformer_block_1 = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)
#     transformer_block_2 = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)
#     transformer_block_3 = TransformerBlock(embed_dim=128, num_heads=num_heads, ff_dim=ff_dim)
#     transformer_block_4 = TransformerBlock(256, num_heads, ff_dim)
#     transformer_block_5 = TransformerBlock(192, num_heads, ff_dim)
#     transformer_block_6 = TransformerBlock(192, num_heads, ff_dim)

#     inputs = Input((2000, 5))
#     x = Conv1D(64, 9, strides=1, padding='same', activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = embedding_layer_1(x)
#     tf1 = transformer_block_1(x)
#     x = Conv1D(64, 9, strides=2, padding='same', activation='relu')(tf1)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = embedding_layer_2(x)
#     tf2 = transformer_block_2(x)
#     x = Conv1D(128, 9, strides=2, padding='same', activation='relu')(tf2)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = embedding_layer_3(x)
#     tf3 = transformer_block_3(x)
#     x = Conv1D(128, 9, strides=1, padding='same', activation='relu')(tf3)
#     x = BatchNormalization()(x)
#     x = LayerNormalization()(x)
#     x = concatenate([x, tf3], axis=-1)
#     x = embedding_layer_4(x)
#     tf4 = transformer_block_4(x)
#     x = Conv1DTranspose(128, kernel_size=9, strides=2, padding='same')(tf4)
#     x = concatenate([x, tf2], axis=-1)
#     x = embedding_layer_5(x)
#     tf5 = transformer_block_5(x)
#     x = Conv1DTranspose(128, kernel_size=9, strides=2, padding='same')(tf5)
#     x = concatenate([x, tf1], axis=-1)
#     x = embedding_layer_6(x)
#     tf6 = transformer_block_6(x)
#     x = Conv1D(128, kernel_size=9, strides=1, padding='same')(tf6)
#     x = Conv1D(64, kernel_size=9, strides=1, padding='same')(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     outputs = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # Vision Transformer
# # adopted from token learner
# # DATA
# BATCH_SIZE = 10
# AUTO = tf.data.AUTOTUNE
# INPUT_SHAPE = (2000, 5)
# NUM_CLASSES = 1

# # OPTIMIZER
# LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-4

# # AUGMENTATION
# IMAGE_SIZE = 2000  # We will resize input images to this size.
# PATCH_SIZE = 4  # Size of the patches to be extracted from the input images.
# NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE)*2

# # ViT ARCHITECTURE
# LAYER_NORM_EPS = 1e-6
# PROJECTION_DIM = 64
# NUM_HEADS = 8
# NUM_LAYERS = 6
# MLP_UNITS = [
#     PROJECTION_DIM/2,
#     PROJECTION_DIM,
# ]

# def mlp(x, dropout_rate, hidden_units):
#     # Iterate over the hidden units and
#     # add Dense => Dropout.
#     for units in hidden_units:
#         x = layers.Dense(units, activation=tf.nn.gelu)(x)
#         x = layers.Dropout(dropout_rate)(x)
#     return x

# def transformer(encoded_patches):
#     # Layer normalization 1.
#     x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)

#     # Multi Head Self Attention layer 1.
#     attention_output = layers.MultiHeadAttention(
#         num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
#     )(x1, x1)

#     # Skip connection 1.
#     x2 = layers.Add()([attention_output, encoded_patches])

#     # Layer normalization 2.
#     x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

#     # MLP layer 1.
#     x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1)

#     # Skip connection 2.
#     encoded_patches = layers.Add()([x4, x2])
#     return encoded_patches

# def position_embedding(projected_patches, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM):
#     # Build the positions.
#     positions = tf.range(start=0, limit=num_patches, delta=1)

#     # Encode the positions with an Embedding layer.
#     encoded_positions = layers.Embedding(
#         input_dim=num_patches, output_dim=projection_dim
#     )(positions)

#     # Add encoded positions to the projected patches.
#     return projected_patches + encoded_positions


# def create_vit_classifier_1D():
#     inputs = layers.Input(shape=INPUT_SHAPE)  # (B, H, W, C)

#     # Create patches and project the pathces.
#     projected_patches = layers.Conv1D(
#         filters=PROJECTION_DIM,
#         kernel_size=(PATCH_SIZE),
#         strides=(2),
#         padding="SAME",
#     )(inputs)
#     _, h, w = projected_patches.shape
#     print(projected_patches.shape)
#     projected_patches = layers.Reshape((h ,w))(
#         projected_patches
#     )  # (B, number_patches, projection_dim)

#     # Add positional embeddings to the projected patches.
#     encoded_patches = position_embedding(
#         projected_patches
#     )  # (B, number_patches, projection_dim)
#     encoded_patches = layers.Dropout(0.1)(encoded_patches)

#     # Iterate over the number of layers and stack up blocks of
#     # Transformer.
#     for i in range(NUM_LAYERS):
#         # Add a Transformer block.
#         encoded_patches = transformer(encoded_patches)

#     # Layer normalization and Global average pooling.
#     representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
#     representation = layers.AveragePooling1D()(representation)
#     representation = layers.Conv1DTranspose(64, kernel_size=4, strides=2, activation='relu', padding='same')(representation)
#     representation = layers.Conv1DTranspose(128, kernel_size=4, strides=2, activation='relu', padding='same')(representation)
#     representation = layers.Dropout(0.2)(representation)
#     representation = layers.Dense(32, activation='relu')(representation)
#     # Classify outputs.
#     outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(representation)

#     # Create the Keras model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


