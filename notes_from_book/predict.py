
import sys , json

label_map_path , emotions_by_path ,  = './label_map.json' , {}

binary_emotions = {

   # bit   # +-
     "0"  :  0 ,
     "1"  :  0 ,
     "2"  :  0 ,
     "3"  :  1 ,

     "4"  :  1 ,
     "5"  :  0 ,
     "6"  :  1
}

# emotions_by_path has the reverse order key,value of label_map_path

# ---------------------------------------------------------------------

with open ( label_map_path , 'r' ) as f : label_map_path = json.load ( f )

                                                     # number_Emotion           # bit
for key in label_map_path.keys () : emotions_by_path [ label_map_path [ key ] ] = key

PRIVATE = '2.CNN/scripts/fer2013'

# clear ; py predict.py 2.CNN/scripts/fer2013/logs/save/model_nfilters_32_kernel_size_5_5_2.h5 2.CNN/scripts/fer2013/PrivateTest | tee new_accuracy.txt

# ---------------------------------------------------------------------

if len ( sys.argv ) > 1: sys.argv = sys.argv [ 1: ]
else:                    sys.argv = []

print ( 'Number of arguments:' , len ( sys.argv ) , 'arguments.' )

if len ( sys.argv ) > 0 :

   print ( 'Model File:' , sys.argv [ 0 ] )

old_corrects = 0
new_corrects = 0

total = 0

# ---------------------------------------------------------------------

from loadModule import predict

if len ( sys.argv ) > 1 :

   print ( 'Images Path:' , sys.argv [ 1 ] )

   for path , value in predict ( sys.argv [ 0 ] , sys.argv [ 1 ] ) :

       if path.count ( PRIVATE ) :
       
          total += 1
          
          parent_path = path.split ( '/' ) [ -1 ]
          
          path_emotion = emotions_by_path [ parent_path ]
          
          path_value_binary = binary_emotions [ path_emotion ]
          pred_value_binary = binary_emotions [      value   ]
          
          if path_emotion      ==      value        : old_corrects += 1
          if path_value_binary == pred_value_binary : new_corrects += 1
       
       else: None

   if sys.argv [ 1 ].count ( PRIVATE ) : print ( '\n' , 'Old ACCURACY =' , str ( old_corrects / total )        )
   if sys.argv [ 1 ].count ( PRIVATE ) : print ( '\n' , 'New ACCURACY =' , str ( new_corrects / total ) , '\n' )

