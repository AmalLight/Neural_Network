
import datetime

print ( str ( datetime.datetime.now () ) ) 

loss = 'categorical_crossentropy'
optimizer = 'adam'

color_channels = 1
img_width, img_height = 48 , 48

img_shape = ( img_width , img_height , color_channels )

epochs     = 100 * 8
batch_size = 512 # if 1000 => 28 steps

#
#    for epoch in epochs :
#
#        for i in range ( int ( total_samples / batch_size ) ) :
#
#            if i is not last : right = batch_size * ( i + 1 )
#            else             : right = total_samples
#
#            doOn ( total_samples [ batch_size * i : right ] )
#

num_classes = 7

nb_train_samples      = 28716 - num_classes
nb_validation_samples =  3596 - num_classes

n_filters    =      32
kernel_size  = ( 5 , 5 )
pooling_size = ( 2 , 2 )

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

import os

basedir = 'fer2013'

logs  = os.path.join ( basedir ,  'logs' )
check = os.path.join (    logs , 'check' )
save  = os.path.join (    logs ,  'save' )

train_feature = os.path.join ( basedir , 'Training'    )
test_feature  = os.path.join ( basedir , 'PrivateTest' )

# processed_images = os.path.join ( basedir , 'processed' )
# train_processed_images = os.path.join ( processed_images , 'Training'    )
# test_processed_images  = os.path.join ( processed_images , 'PrivateTest' )

for folder in [ basedir , logs , check , save , train_feature , test_feature ] :

    if not os.path.exists ( folder ):   os.makedirs ( folder )

    else: print ( 'Folder {} exists already'.format ( folder ) )

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

from keras.models               import Sequential
from keras.layers               import Dense, Dropout , Flatten
from keras.layers.convolutional import Convolution2D  , MaxPooling2D

model_name = 'model_nfilters_' + str ( n_filters ) + '_kernel_size_' + str ( kernel_size [ 0 ] ) + '_' + str ( kernel_size [ 1 ] )

model = Sequential ( name = ( model_name ) )

model.add ( Convolution2D ( n_filters , kernel_size , padding = 'same' , input_shape = img_shape , activation = 'relu' ) )

model.add ( MaxPooling2D ( pool_size = pooling_size , padding = 'same' ) )

model.add ( Convolution2D ( n_filters , kernel_size , activation = 'relu' , padding = 'same' ) )

model.add ( MaxPooling2D ( pool_size = pooling_size , padding = 'same' ) )

model.add ( Dropout ( 0.2 ) )
model.add ( Flatten () )
model.add ( Dense ( 128 , activation = 'relu' ) )
model.add ( Dense ( num_classes , activation ='softmax' , name = 'preds' ) )

model.compile ( loss = loss , optimizer = optimizer , metrics = [ 'accuracy' ] )

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator (

    rescale = 1. / 255 ,
    shear_range  = 0.2 ,
    zoom_range   = 0.2 ,

    featurewise_center            = True ,
    featurewise_std_normalization = True ,

    rotation_range    =  20 ,
    width_shift_range = 0.2 ,

    height_shift_range =  0.2 ,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator ( rescale = 1. / 255 ) # rescale != resize => to get little values != to get black and white

# Use folders to get data and labels
# Setting the train_generatorand validation_generator to categorical and it will one-hot encode your classes.

train_generator = train_datagen.flow_from_directory ( # get from_directory and evalues it using the dir's name?

    directory = train_feature ,

    color_mode = 'grayscale' ,

    target_size = ( img_width , img_height ) ,

    batch_size = batch_size ,

    class_mode='categorical' ,

    # save_to_dir=train_processed_images
)

validation_generator = test_datagen.flow_from_directory ( # get from_directory and tests it using the dir's name?

    directory = test_feature ,

    color_mode = 'grayscale' ,

    target_size = ( img_width , img_height ) ,

    batch_size = batch_size ,

    class_mode = 'categorical' ,

    # save_to_dir = test_processed_images
)

import parentScripts.callbacks_list as callbacks_list

callbacks_list = callbacks_list.lista ( logs , check )

nn_history = model.fit_generator (

    train_generator ,
    steps_per_epoch = nb_train_samples // batch_size ,
    epochs = epochs ,

    validation_data = validation_generator ,
    validation_steps = nb_validation_samples // batch_size ,
    
    callbacks = callbacks_list ,
    workers = 4
)

model.save ( os.path.join ( save , model.name ) + ".h5" )

print ( "model saved on:" , os.path.join ( save , model.name ) + ".h5" )

print ( str ( datetime.datetime.now () ) )
