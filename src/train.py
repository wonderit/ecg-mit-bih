from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from graph_small import ECG_model
from config import get_config
from utils import *

def train(config, X, y, Xval=None, yval=None):
    
    classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
    # expand dims
    Xe = np.expand_dims(X, axis=2)
    if not config.split:
        from sklearn.model_selection import train_test_split
        Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.2, random_state=1)
    else:
        #  expand dims
        Xvale = np.expand_dims(Xval, axis=2)
        # (m, n) = y.shape
        # y = y.reshape((m, 1, n ))
        # (mvl, nvl) = yval.shape
        # yval = yval.reshape((mvl, 1, nvl))

    if config.checkpoint_path is not None:
        model = model.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch # put the resuming epoch
    else:
        model = ECG_model(config)
        initial_epoch = 0

    mkdir_recursive('models')
    #lr_decay_callback = LearningRateSchedulerPerBatch(lambda epoch: 0.1)
    callbacks = [
            EarlyStopping(patience = config.patience, verbose=1),
            ReduceLROnPlateau(factor = 0.5, patience = 3, min_lr = 0.01, verbose=1),
            TensorBoard( log_dir='./logs', histogram_freq=0, write_graph = True, write_grads=False, write_images=True),
            ModelCheckpoint('models/{}-small-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, period=10)
            # , lr_decay_callback
    ]
    data_dir = 'processed_data'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_file_suffix = 'train'
    test_file_suffix = 'test'

    file_name_train_x = 'X{}'.format(train_file_suffix)
    file_name_train_y = 'y{}'.format(train_file_suffix)
    file_name_test_x = 'X{}'.format(test_file_suffix)
    file_name_test_y = 'y{}'.format(test_file_suffix)
    train_x = np.array(X)
    train_y = np.array(y)
    test_x = np.array(Xval)
    test_y = np.array(yval)
    print('train_x shape : ', train_x.shape)
    print('train_y shape : ', train_y.shape)
    print('test_x shape : ', test_x.shape)
    print('test_y shape : ', test_y.shape)
    # np.savetxt('{}/{}'.format(data_dir, file_name_train_x), train_x, delimiter=',', fmt='%1.8f')
    # np.savetxt('{}/{}'.format(data_dir, file_name_train_y), train_y, delimiter=',', fmt='%1.8f')
    # np.savetxt('{}/{}'.format(data_dir, file_name_test_x), test_x, delimiter=',', fmt='%1.8f')
    # np.savetxt('{}/{}'.format(data_dir, file_name_test_y), test_y, delimiter=',', fmt='%1.8f')

    model.fit(Xe, y,
            validation_data=(Xvale, yval),
            epochs=config.epochs,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)
    print_results(config, model, Xvale, yval, classes, )

    #return model

def main(config):
    print('feature:', config.feature)
    #np.random.seed(0)
    (X,y, Xval, yval) = loaddata(config.input_size, config.feature)
    train(config, X, y, Xval, yval)

if __name__=="__main__":
    config = get_config()
    main(config)
