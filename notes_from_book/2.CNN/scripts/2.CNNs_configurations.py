from keras.datasets import mnist

# the data, split between train and test sets
( X_train , y_train ) , ( X_test , y_test ) = mnist.load_data ()

# input image dimensions
img_rows , img_cols = X_train [ 0 ].shape

# Reshaping the data to use it in our network
X_train = X_train.reshape ( X_train.shape [ 0 ] , img_rows , img_cols , 1 )
X_test  = X_test.reshape  ( X_test.shape  [ 0 ] , img_rows , img_cols , 1 )
input_shape =                                   ( img_rows , img_cols , 1 )

# Scaling the data
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# import keras
# convert class vectors to binary class matrices
# num_classes = 10
# y_train = keras.utils.to_categorical ( y_train , num_classes ) # old: 'keras.utils' has no attribute 'to_categorical'
# y_test  = keras.utils.to_categorical ( y_test  , num_classes ) # old: 'keras.utils' has no attribute 'to_categorical'

loss = 'categorical_crossentropy'
optimizer = 'adam'

N_SAMPLES = 30_000

X_train = X_train [ : N_SAMPLES ]
X_test  = X_test  [ : N_SAMPLES ]
y_train = y_train [ : N_SAMPLES ]
y_test  = y_test  [ : N_SAMPLES ]

from sklearn.preprocessing import LabelBinarizer
label_as_binary = LabelBinarizer () # convert output ( vectors ) to binary class matrices

y_labels_train = label_as_binary.fit_transform ( y_train )
y_labels_test  = label_as_binary.fit_transform ( y_test  )

filters      = [ 4 , 8 , 16 ]                                      # count 3 , filters are units?
kernal_sizes = [ ( 2 , 2 ) , ( 3 , 3 ) , ( 4 , 4 ) , ( 16 , 16 ) ] # count 4

import itertools , os

config = itertools.product ( filters, kernal_sizes) # [ ( x , y ) for x in A for y in B ]

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPooling2D

model_directory = 'models'

epochs , batch_size = 1 , 512

num_classes = 10

for n_filters , kernel_size in config :

    model_name = 'single_f_' + str ( n_filters ) + '_k_' + str ( kernal_sizes.index ( kernel_size ) )
    
    model = Sequential ( name = model_name )

    model.add ( Conv2D ( n_filters , kernel_size=kernel_size , activation='relu' , input_shape = input_shape ) )

    model.add ( Flatten () )
    model.add ( Dense ( num_classes , activation = 'softmax' ) )
    
    model.compile ( loss = loss , optimizer = optimizer , metrics = [ 'accuracy' ] )
    model.fit ( X_train , y_labels_train , batch_size = batch_size , epochs = epochs , verbose = 1 , validation_data = ( X_test , y_labels_test ) )
    
    score = model.evaluate ( X_test , y_labels_test , verbose = 0 )

    print ( 'Test loss: {} - Test accuracy: {}'.format ( score [ 0 ] , score [ 1 ] ) )

    model_path = os.path.join ( model_directory , model_name )
    model.save ( model_path )
    
    print ( 'model saved on:' , model_path )

