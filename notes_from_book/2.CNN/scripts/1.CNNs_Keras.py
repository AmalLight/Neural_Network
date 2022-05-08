from keras.datasets import mnist

# the data, split between train and test sets
( X_train , y_train ) , ( X_test , y_test ) = mnist.load_data ()

# input image dimensions
img_rows, img_cols = X_train [ 0 ].shape

# Reshaping the data to use it in our network
X_train = X_train.reshape ( X_train.shape [ 0 ] , img_rows , img_cols , 1 )
X_test  = X_test.reshape  ( X_test.shape  [ 0 ] , img_rows , img_cols , 1 )
input_shape =             (                       img_rows , img_cols , 1 )

# Scaling the data
X_train = X_train / 255.0 # Greys
X_test  = X_test  / 255.0 # Greys

import numpy as np
from matplotlib import pyplot as plt

plt.imshow ( X_test [ 1 ] [ ... , 0 ] , cmap='Greys' )
plt.axis ( 'off' )
# plt.show ()

# sudo ln -s /home/kaumi/Git/Neural_Network/notes_from_book ./parentScripts
import parentScripts.saveFigure as saveFigure

saveFigure.save ( plt , "1.Number_from_Mnist" )

# ------------------------------------------------------------------------------

from keras.models   import Sequential
from keras.layers   import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

batch_size = 128
epochs = 2

model = Sequential ()

# activation itself is not convolution, example soft max [ 1 , 1 , 3 ] => [ 0.1 , 0.1 , 0.8 ] ( no reduction )

# X * W1 | X = 10x10 and W1 = 10x5 == 10x5, this is not convolution ( result is not a sub square ),
# AFTER, X.T = 5x10 * W2 | W2 = 10x5 = 5x5.

# I don't know if I can make a Max_pooling alghoritm using only this two anonymous W1 & W2. W1 & W2 can be only contained in at least of two layers.

model.add ( Conv2D ( 32 , kernel_size = ( 3 , 3 ) , activation = 'relu' , input_shape = input_shape ) )
model.add ( Conv2D ( 32 ,               ( 3 , 3 ) , activation = 'relu'                             ) )

# As you can see by the saved pictures 3x3 is the most famous Kernel's size for these kind of operations
# ( the filter + kernel moltiplication ) is equal to do same steps with W1 and W2, but in once round.
# I think the kernel's numbers ( inside itself ), are chosen randomly, right?

model.add ( MaxPooling2D ( pool_size = ( 2 , 2 ) ) ) # without kernel, it's only a one convolution operation

model.add ( Dropout ( 0.25 ) )

model.add ( Flatten () ) # matrix to vector ?

model.add ( Dense ( 128 , activation='relu' ) )

model.add ( Dropout ( 0.3 ) )

# We know we have 10 classes
# which are the digits from 0-9
num_classes = 10

model.add ( Dense ( num_classes , activation = 'softmax' ) ) # it seems that softmax was made for all vertical hidden layer
                                                             # are activations made for each nodes or for all hidden layer nodes?
"""

-O --|--o
     |   \
     |    [A] | A = 1 0 0 0 0 0 ?
     |   /
-O --|--o
     |   \
     |    [B] | B = 0 1 0 0 0 0 ?
     |   /
-O --|--o
     |   \
     |    [C] | C = 0 0 1 0 0 0 ?
     |   /
-O --|--o
     |   \
     |    [D] ..
     |   /
-O --|--o
     |   \
     |    [E] ..
     |   /
-O --|--o
     |   \
     |    [F] ..
     |   /
-O --|--o

softmax could be equal to Softmax ( sum ( X * W ) ), where W is X * A + X * B ... Without bias.

Only if Flatten () is matrix to vector.

I think that for this softmax there is an exception, it can't be for the previous layers.
or we had:

-------------->

 HL1        HL2

          _/
-O-\      | -O
-O--\     | -O
-O---[@]--| -O
-O---/    | -O
-O--/     | -O
          ^\
-------------->

or maybe softmax created an implicit a matrix made like this: [ 1 0 0 ; 0 1 0 ; 0 0 1 ] = W | Softmax ( sum ( X * W ) )

"""

loss = 'categorical_crossentropy'
optimizer = 'adam'

model.compile ( loss = loss , optimizer = optimizer , metrics = [ 'accuracy' ] )

# model.fit ( X_train , y_train , batch_size = batch_size , epochs = epochs , verbose = 1 , validation_data = ( X_test , y_test ) )

from sklearn.preprocessing import LabelBinarizer

label_as_binary = LabelBinarizer () # convert output ( number? ) to classifications

# if out == 1: 0 0 1 # A
# if out == 2: 0 1 0 # B
# if out == 3: 0 1 1 # C

y_labels_train = label_as_binary.fit_transform ( y_train )
y_labels_test  = label_as_binary.fit_transform ( y_test  )

model.fit ( X_train , y_labels_train , batch_size = batch_size , epochs = epochs , verbose = 1 , validation_data = ( X_test , y_labels_test ) )

score = model.evaluate ( X_test , y_labels_test , verbose = 0 )

print ( 'Test loss: {} - Test accuracy: {}'.format ( score [ 0 ] , score [ 1 ] ) )

