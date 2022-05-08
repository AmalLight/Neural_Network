img_width, img_height = 48 , 48

# ---------------------------------------------------------------------

processed = '/home/kaumi/Git/Neural_Network/notes_from_book/3.predicts/processed'

import os , json

label_map_path , emotions = './label_map.json' , {}

with open ( label_map_path , 'r' ) as f : emotions = json.load ( f )

# ---------------------------------------------------------------------

def create_folder ( path , show ) :

    if not os.path.exists ( path ) : os.makedirs ( path )
    else:

       if not path in show :

          print ( 'Path {} exists already'.format ( path ) )

          show [ path ] = True

    return show

create_folder ( processed , {} )

# ---------------------------------------------------------------------

newsize = ( img_width , img_height )

import tensorflow as tf

from keras.preprocessing import image

import numpy as np

def predict ( model_file_hdf5 , img_path_to_predict ) :

    count , show = 0 , {}

    loaded_model = tf.keras.models.load_model ( model_file_hdf5 )
    
    print ( '' )

    for path , dirs , files in os.walk ( img_path_to_predict ) :
    
        if path.count ( '3.predicts/processed' ) : continue

        for img_file in files :
        
            count += 1

            full_img_file =  os.path.join ( path , img_file )
            
            processed_parent = os.path.join ( processed , path.split ( '/' ) [ -1 ] )
    
            img = image.load_img ( full_img_file , target_size = ( img_width , img_height ) , color_mode = "grayscale" )

            show = create_folder ( processed_parent , show )

            # img.save ( os.path.join ( processed_parent , img_file ) )
            
            img = image.img_to_array ( img ).astype ( 'float32' ) / 255
            
            img = np.expand_dims ( img , axis = 0 )

            predicted = np.argmax ( loaded_model.predict ( img ) , axis = -1 )
            
            value = str ( predicted [ 0 ] )

            emotion = emotions [ value ] [ 2 : ]

            print ( '(' , count , ')' , 'done for:' , full_img_file  , ':' , emotion )

            yield path , value

