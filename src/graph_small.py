from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization, Lambda
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model

def ECG_model(config):
    """ 
    implementation of the model in https://www.nature.com/articles/s41591-018-0268-3 
    also have reference to codes at 
    https://github.com/awni/ecg/blob/master/ecg/network.py 
    and 
    https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/train_model.py
    """
    # def first_conv_block(inputs, config):
    #     layer = Conv1D(filters=config.filter_length,
    #            kernel_size=config.kernel_size,
    #            padding='same',
    #            strides=1,
    #            kernel_initializer='he_normal')(inputs)
    #     layer = BatchNormalization()(layer)
    #     layer = Activation('relu')(layer)
    #
    #     shortcut = MaxPooling1D(pool_size=1,
    #                   strides=1)(layer)
    #
    #     layer =  Conv1D(filters=config.filter_length,
    #            kernel_size=config.kernel_size,
    #            padding='same',
    #            strides=1,
    #            kernel_initializer='he_normal')(layer)
    #     layer = BatchNormalization()(layer)
    #     layer = Activation('relu')(layer)
    #     layer = Dropout(config.drop_rate)(layer)
    #     layer =  Conv1D(filters=config.filter_length,
    #                     kernel_size=config.kernel_size,
    #                     padding='same',
    #                     strides=1,
    #                     kernel_initializer='he_normal')(layer)
    #     return add([shortcut, layer])
    #
    # def main_loop_blocks(layer, config):
    #     filter_length = config.filter_length
    #     n_blocks = 2
    #     for block_index in range(n_blocks):
    #         def zeropad(x):
    #             """
    #             zeropad and zeropad_output_shapes are from
    #             https://github.com/awni/ecg/blob/master/ecg/network.py
    #             """
    #             y = K.zeros_like(x)
    #             return K.concatenate([x, y], axis=2)
    #
    #         def zeropad_output_shape(input_shape):
    #             shape = list(input_shape)
    #             assert len(shape) == 3
    #             shape[2] *= 2
    #             return tuple(shape)
    #
    #         subsample_length = 2 if block_index % 2 == 0 else 1
    #         shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    #
    #         # 5 is chosen instead of 4 from the original model
    #         if block_index % 4 == 0 and block_index > 0 :
    #             # double size of the network and match the shapes of both branches
    #             shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    #             filter_length *= 2
    #
    #         layer = BatchNormalization()(layer)
    #         layer = Activation('relu')(layer)
    #         layer =  Conv1D(filters= filter_length,
    #                         kernel_size=config.kernel_size,
    #                         padding='same',
    #                         strides=subsample_length,
    #                         kernel_initializer='he_normal')(layer)
    #         layer = BatchNormalization()(layer)
    #         layer = Activation('relu')(layer)
    #         layer = Dropout(config.drop_rate)(layer)
    #         layer =  Conv1D(filters= filter_length,
    #                         kernel_size=config.kernel_size,
    #                         padding='same',
    #                         strides= 1,
    #                         kernel_initializer='he_normal')(layer)
    #         layer = add([shortcut, layer])
    #     return layer
    #
    # def output_block(layer, config):
    #     from keras.layers.wrappers import TimeDistributed
    #     layer = BatchNormalization()(layer)
    #     layer = Activation('relu')(layer)
    #     #layer = Flatten()(layer)
    #     outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
    #     model = Model(inputs=inputs, outputs=outputs)
    #
    #     adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #     model.compile(optimizer= adam,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #     model.summary()
    #     return model

    classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S'] are too few or not in the trainset, so excluded out
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 1), name='input')
    # layer = first_conv_block(inputs, config)
    # layer = main_loop_blocks(layer, config)
    # return output_block(layer, config)

    #TODO ADD DROPOUT
    ## First convolutional block (conv,BN, relu)

    x = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(inputs)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    # x = Dropout(config.drop_rate)(x)
    x = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    # x = Dropout(config.drop_rate)(x)
    # similar implementation to maxpool
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)
    ## 2 convolutional block (conv,BN, relu)
    x = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    # x = Dropout(config.drop_rate)(x)
    ## 3 convolutional block (conv,BN, relu)
    x = Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(config.drop_rate)(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # filter size : 32, filter length : 16
    # w/o drop out : 0.83
    # w dropout : 0.89
    # after flatten : 0.86
    # similar implementation to maxpool
    # x = Dropout(config.drop_rate)(x)

    # filter size : 16, filter length : 7
    # w/o drop out : 0.82
    # w dropout : 0.89
    # after flatten : 0.80
    # all dropout : 0.91
    # 1 dropout : 0.89
    # x = Dropout(config.drop_rate)(x)

    # Final bit
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Flatten()(x)
    # x = Dropout(config.drop_rate)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(config.drop_rate_large)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(config.drop_rate_large)(x)

    out = Dense(len_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    # sequential_model_to_ascii_printout(model)
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
