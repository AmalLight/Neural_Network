import numpy  as np
import pandas as pd

np.random.seed ( 11 )

from keras.models      import Sequential
from keras.layers.core import Dense, Dropout, Activation
 
from keras.callbacks   import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

from sklearn.metrics import roc_auc_score , mean_squared_error , confusion_matrix

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

import callbacks_list

callbacks_list = callbacks_list.lista ( '.' )

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

#### Creating the dataset

mu_x1, sigma_x1 = 0, 0.1

x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

                            # generates vect # start #   end    #  set
d1 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 0,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 0, 'type': 0 } )

d2 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 1,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 0, 'type': 1 } )

d3 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 0,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 1, 'type': 0 } )

d4 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) - 1,
                      'x2': np.random.normal ( mu_x1 , sigma_x1 , 1000 ) + 1, 'type': 1 } )

# variables are columns, rows are the datasets and each row is one completed input, you must remember R

data = pd.concat ( [d1, d2, d3, d4] , ignore_index=True ) # concat of rows

print ( 'print len data row:' , len ( data             ) ) # 4000
print ( 'print len data col:' ,       data.shape [ 1 ] )   # 3, third is type

print ( 'example x1: ', data [  'x1'  ] [ 5 ] ) # 0.
print ( 'example x2: ', data [  'x2'  ] [ 5 ] ) # 0.
print ( 'example  3: ', data [ 'type' ] [ 5 ] ) # 0

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8 # test will be +- 20% == +- 800

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1', 'x2']][msk], data[['type']][msk].values # train will be +- 80% == +- 3200

# Everything else will go into the validation set
test_x, test_y = data[['x1', 'x2']][~msk], data[['type']][~msk].values

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

model = Sequential ()

# in Keras model.add Dense defines only one vertical layer,
# units are implicit and not defined as units = 2, but only 2.
# here all unit learns from 2 input ( in this case inputs are columns )

                      # input of 2 neurons
model.add ( Dense ( 2 , input_dim = 2 ) )

model.add ( Activation ( 'tanh' ) ) # W is not specifed, there is ? there are W1 and/or W2 ?

model.add ( Dense ( 1 ) ) # 1 is the number of last neurons/units in output, input_dim ?

model.add ( Activation ( 'sigmoid' ) ) # Wout is not specifed, there is ?

# FINAL GRAPH ------
        _
X1 --- (_)
  ^\  /^ \     _
    \/    \>> (_) ----> Predict!
  __/\_ _ /
X2 --- (_)

# END --------------

from tensorflow.keras.optimizers  import SGD

sgd = SGD ( learning_rate = 0.1 )
model.compile ( loss = 'mse' , optimizer = sgd , metrics = [ 'accuracy' ] )

model.fit ( train_x [ [ 'x1' , 'x2' ] ] , train_y , batch_size=1 , epochs=2 , callbacks = callbacks_list )

pred = model.predict ( test_x )
print ( 'MSE:' , mean_squared_error ( test_y , pred ) )

