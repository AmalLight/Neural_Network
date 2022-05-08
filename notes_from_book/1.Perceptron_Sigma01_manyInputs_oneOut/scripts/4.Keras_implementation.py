# from keras.layers import Activation, Dense
# model.add ( Dense ( 32 ) )
# model.add ( Activation ( 'tanh' ) )

# This is equivalent to the following command: model.add ( Dense ( 32 , activation = 'tanh' ) )
# You can also pass an element-wise TensorFlow/Theano/CNTK function as an activation:

# from keras import backend as K
# model.add ( Dense (32 , activation = K.tanh ) )

import numpy  as np
import pandas as pd

np.random.seed ( 11 )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#### Creating the dataset
# mean and standard deviation for the x belonging to the first class

mu_x1, sigma_x1 = 0, 0.1

# Constant to make the second distribution different from the first

x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

# x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5

# creating the first distribution

d1 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1, sigma_x1, 1000 ) + 0,
                      'x2': np.random.normal ( mu_x1, sigma_x1, 1000 ) + 0, 'type': 0 } )

d2 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1, sigma_x1, 1000 ) + 1,
                      'x2': np.random.normal ( mu_x1, sigma_x1, 1000 ) - 0, 'type': 1 } )

d3 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1, sigma_x1, 1000 ) - 0,
                      'x2': np.random.normal ( mu_x1, sigma_x1, 1000 ) - 1, 'type': 0 } )

d4 = pd.DataFrame ( { 'x1': np.random.normal ( mu_x1, sigma_x1, 1000 ) - 1,
                      'x2': np.random.normal ( mu_x1, sigma_x1, 1000 ) + 1, 'type': 1 } )

data = pd.concat ( [d1, d2, d3, d4] , ignore_index=True )

import seaborn as sns; sns.set ()

import saveFigure

ax = sns.scatterplot ( x="x1" , y="x2" , hue="type" , data=data )

saveFigure.save ( ax , '4.Keras_implementation_d1_d2_d3_d4_distributions' )
