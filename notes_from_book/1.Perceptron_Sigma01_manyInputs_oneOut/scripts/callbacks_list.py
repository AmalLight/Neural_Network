from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

import os

def lista ( path ):

    logs = os.path.join ( '.' , 'logs' )

    tbCallBack = TensorBoard ( log_dir=logs , histogram_freq=0 , write_graph=True , write_images=True )

    callbacks_list = [ tbCallBack ]
    
    filepath = "checkpoint-{epoch:02d}-{accuracy:.2f}.hdf5"

    checkpoint = ModelCheckpoint ( filepath , monitor = 'accuracy' , verbose = 1 , save_best_only = False , mode = 'max' )
    
    callbacks_list += [ checkpoint ]
    
    return callbacks_list

# tensorboard --logdir ./logs
