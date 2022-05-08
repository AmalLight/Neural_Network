import numpy as np
from PIL import Image

def row2image ( pixels ) : # https://www.kaggle.com/an1meshjain/facial-expression-classification

    img_array = np.array ( pixels.split () )
    
    img_array = np.reshape ( img_array , ( 48 , 48 ) )

    return img_array.astype ( np.uint8 )


def save_image ( row_number , pixels , file_name ) :

    try :

        im = Image.fromarray ( row2image ( pixels ) )
        im.save (  file_name )
        
        print ( '(' , row_number , ')' , 'saved image:' , file_name )

    except e : print ( e )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import os , json

import pandas as pd

# Pixel values have range from 0 to 255 ( 0 is black and 255 is white )

# https://www.kaggle.com/deadskull7/fer2013 --- csv with images inside ( pixel values )

import parentScripts.readFileCSV as readFileCSV

basedir = 'fer2013'

freCSVtext = readFileCSV.read ( basedir , 'file' , 36000 , False ) # ALL are : 35887

data_input = pd.DataFrame ( [ x.split ( ',' ) for x in freCSVtext.split ( '\n' ) ] , columns = [ 'emotion' , 'pixels' , 'Usage' ] )

data_input.head ()

label_map_path , label_map = './parentScripts/label_map_2013.json' , {}

with open ( label_map_path , 'r' ) as f : label_map = json.load ( f )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Creating the folders

output_folders = data_input [ 'Usage' ] .unique () .tolist ()

while ( None in output_folders ) : output_folders.remove ( None )

print ( 'output_folders:' , output_folders )

all_folders = []

for folder in output_folders:

    for label in label_map:

        all_folders.append ( os.path.join ( basedir , folder , label_map [ label ] ) )

for folder in all_folders:

    if not os.path.exists ( folder ):   os.makedirs ( folder )

    else: print ( 'Folder {} exists already'.format ( folder ) )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# print ( data_input [ 'pixels' ] [ 0 ] )

# import sys

# sys.exit ()

Usage_dict = {}

data_input.head ()

row_number = 1

for _ , row in data_input.iterrows () :
    
    folder = row.Usage

    if folder == None : continue
    
    emotion = row.emotion

    key = folder + emotion

    if not key in Usage_dict: Usage_dict [ key ] =                      1
    else:                     Usage_dict [ key ] = Usage_dict [ key ] + 1

    index_Usage = Usage_dict [ key ]
    
    if ( index_Usage >= 1 ) and ( emotion in label_map ) :

         file_name = os.path.join ( basedir , folder , label_map [ emotion ] ) + '/image_' + str ( index_Usage ) + '.png'

         save_image ( row_number , row.pixels , file_name )
         
         row_number += 1
         
    else :

         print ( 'error if index_Usage <= 0:' , index_Usage )

         print ( 'error if not emotion:' , emotion , 'in label_map:' , str ( emotion in label_map ) )

